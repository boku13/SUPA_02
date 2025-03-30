import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrained_eval import load_pretrained_model, calculate_eer, calculate_tar_at_far
from finetune import SpeakerVerificationModel, apply_lora, get_embedding_dim, verify_lora_activation, get_target_modules
from utils import load_audio, setup_seed
from peft import LoraConfig, get_peft_model, TaskType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def diagnose_model_weights(model_path):
    """Examine the saved model weights"""
    logger.info(f"Examining saved model: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    logger.info(f"State dict contains {len(state_dict)} keys")
    
    # Count parameters containing 'lora'
    lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
    logger.info(f"Found {len(lora_keys)} keys containing 'lora'")
    
    # Check values of LoRA parameters
    nonzero_lora = 0
    for k in lora_keys[:10]:  # Check first 10 LoRA keys
        param = state_dict[k]
        param_sum = torch.abs(param).sum().item()
        param_norm = torch.norm(param).item()
        
        logger.info(f"LoRA param {k}: sum={param_sum:.6f}, norm={param_norm:.6f}")
        if param_sum > 1e-6:
            nonzero_lora += 1
    
    if nonzero_lora == 0 and len(lora_keys) > 0:
        logger.error("All examined LoRA parameters have approximately zero values!")
        logger.error("This suggests the LoRA weights weren't properly trained or saved.")
    
    return nonzero_lora > 0

def compare_embeddings(model_name, finetuned_model_path, audio_file):
    """Compare embeddings from pretrained, finetuned (dynamic LoRA), and merged LoRA models"""
    logger.info(f"Comparing embeddings using {model_name} on {audio_file}")
    
    # --- 1. Load Pretrained Model & Audio ---
    pretrained_model, processor = load_pretrained_model(model_name)
    pretrained_model.eval()
    
    waveform, sr = load_audio(audio_file)
    inputs = processor(
        waveform.numpy(), 
        sampling_rate=sr, 
        return_tensors="pt"
    ).input_values.to(device)
    
    # --- 2. Get Embedding from Pretrained Model ---
    with torch.no_grad():
        pretrained_outputs = pretrained_model(inputs)
        pretrained_emb = pretrained_outputs.last_hidden_state.mean(dim=1)
        pretrained_emb = torch.nn.functional.normalize(pretrained_emb, p=2, dim=1)
        logger.info(f"Pretrained Embedding Norm: {torch.norm(pretrained_emb).item():.6f}")

    # --- 3. Create and Load Finetuned Model (Dynamic LoRA) ---
    # Need a fresh base model instance for the finetuned version
    base_model_ft, _ = load_pretrained_model(model_name) 
    base_model_ft.eval() # Keep it in eval mode
    
    embedding_dim = get_embedding_dim(model_name)
    # NOTE: Wrap the fresh base_model_ft here
    finetuned_model = SpeakerVerificationModel(
        base_model_ft, 
        embedding_dim,
        100 # Assuming 100 speakers used during training
    ).to(device)
    
    # Apply LoRA structure
    finetuned_model = apply_lora(finetuned_model, model_name) # apply_lora enables adapters
    
    # Load trained weights
    logger.info(f"Loading weights from {finetuned_model_path} into dynamic LoRA model")
    try:
        state_dict = torch.load(finetuned_model_path, map_location=device)
        # Load with strict=False as state_dict might contain extra keys (like arcface)
        finetuned_model.load_state_dict(state_dict, strict=False) 
        logger.info("Weights loaded successfully into dynamic LoRA model.")
    except Exception as e:
        logger.error(f"Error loading weights into dynamic LoRA model: {e}")
        return False, False # Indicate failure

    # Ensure LoRA adapters are active (apply_lora should handle this, but double check)
    verify_lora_activation(finetuned_model)
    finetuned_model.eval()

    # --- 4. Get Embedding from Finetuned Model (Dynamic LoRA) ---
    with torch.no_grad():
        # Use the backbone directly
        backbone_outputs_ft = finetuned_model.backbone(inputs) 
        finetuned_emb_dynamic = backbone_outputs_ft.last_hidden_state.mean(dim=1)
        finetuned_emb_dynamic = torch.nn.functional.normalize(finetuned_emb_dynamic, p=2, dim=1)
        logger.info(f"Finetuned (Dynamic) Embedding Norm: {torch.norm(finetuned_emb_dynamic).item():.6f}")
        
    # --- 5. Create, Load, and Merge LoRA weights ---
    # Need another fresh base model instance for the merged version
    base_model_merged, _ = load_pretrained_model(model_name)
    base_model_merged.eval() # Keep it in eval mode
    
    # Apply LoRA structure AGAIN to this fresh base model
    # Use the same PeftConfig as before
    target_modules = get_target_modules(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, 
        lora_dropout=0.1, target_modules=target_modules, bias="none", 
        inference_mode=False, init_lora_weights="gaussian" 
    )
    merged_model = get_peft_model(base_model_merged, lora_config)

    # Load ONLY the LoRA weights from the saved state_dict
    logger.info(f"Loading weights from {finetuned_model_path} into model before merging")
    try:
        state_dict = torch.load(finetuned_model_path, map_location=device)
        # Load ONLY LoRA weights - PEFT model expects keys without 'base_model.model.' prefix
        # We need to adapt the keys from the saved state_dict
        adapted_state_dict = {}
        prefix_to_strip = "base_model.model." 
        for key, value in state_dict.items():
             if "lora_" in key: # Only load LoRA parameters
                 new_key = key.replace(prefix_to_strip, "")
                 adapted_state_dict[new_key] = value
        
        missing_keys, unexpected_keys = merged_model.load_state_dict(adapted_state_dict, strict=False)
        logger.info(f"LoRA weights loaded for merging. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
        if any("lora" in k for k in missing_keys):
             logger.warning(f"Some LoRA weights seem to be missing during merge load: {missing_keys}")
        
    except Exception as e:
        logger.error(f"Error loading weights for merging: {e}")
        return False, False # Indicate failure

    # Merge the weights
    logger.info("Merging LoRA weights...")
    try:
        # merged_model = merged_model.merge_and_unload() # Returns the base model with merged weights
        # TEMP FIX: merge_and_unload might have issues, try manual merge logic if needed
        # For now, assume merge_and_unload works as intended by PEFT
        merged_model = merged_model.merge_and_unload()
        logger.info("LoRA weights merged successfully.")
        merged_model.eval()
    except Exception as e:
        logger.error(f"Error merging LoRA weights: {e}")
        return False, False # Indicate failure

    # --- 6. Get Embedding from Merged Model ---
    with torch.no_grad():
        merged_outputs = merged_model(inputs)
        finetuned_emb_merged = merged_outputs.last_hidden_state.mean(dim=1)
        finetuned_emb_merged = torch.nn.functional.normalize(finetuned_emb_merged, p=2, dim=1)
        logger.info(f"Finetuned (Merged) Embedding Norm: {torch.norm(finetuned_emb_merged).item():.6f}")

    # --- 7. Comparisons ---
    diff_dynamic = torch.norm(pretrained_emb - finetuned_emb_dynamic).item()
    cos_sim_dynamic = torch.sum(pretrained_emb * finetuned_emb_dynamic, dim=1).item()
    
    diff_merged = torch.norm(pretrained_emb - finetuned_emb_merged).item()
    cos_sim_merged = torch.sum(pretrained_emb * finetuned_emb_merged, dim=1).item()

    logger.info(f"--- Dynamic LoRA Comparison ---")
    logger.info(f"Embedding difference (L2 norm): {diff_dynamic:.6f}")
    logger.info(f"Embedding similarity (cosine): {cos_sim_dynamic:.6f}")
    
    logger.info(f"--- Merged LoRA Comparison ---")
    logger.info(f"Embedding difference (L2 norm): {diff_merged:.6f}")
    logger.info(f"Embedding similarity (cosine): {cos_sim_merged:.6f}")

    embeddings_differ_dynamic = diff_dynamic > 1e-6
    embeddings_differ_merged = diff_merged > 1e-6

    if not embeddings_differ_dynamic:
        logger.error("CRITICAL (Dynamic): Embeddings are identical! Dynamic LoRA is not being applied during inference.")
    else:
        logger.info("SUCCESS (Dynamic): Dynamic LoRA embeddings are different from pretrained.")
        
    if not embeddings_differ_merged:
        logger.error("CRITICAL (Merged): Embeddings after merging are identical! LoRA weights are ineffective or merge failed.")
    else:
        logger.info("SUCCESS (Merged): Merged LoRA embeddings are different from pretrained.")

    return embeddings_differ_dynamic, embeddings_differ_merged

def evaluate_trial_pairs(model_name, finetuned_model_path, trial_file, audio_root, max_trials=10):
    """Evaluate a small number of trial pairs to verify LoRA impact"""
    logger.info(f"Evaluating {max_trials} trial pairs using {model_name}")
    
    # Load pretrained model
    pretrained_model, processor = load_pretrained_model(model_name)
    pretrained_model.eval()
    
    # Create finetuned model
    embedding_dim = get_embedding_dim(model_name)
    finetuned_model = SpeakerVerificationModel(
        pretrained_model,
        embedding_dim,
        100  # 100 speakers used during training
    ).to(device)
    
    # Apply LoRA
    finetuned_model = apply_lora(finetuned_model, model_name)
    
    # Load weights
    try:
        state_dict = torch.load(finetuned_model_path, map_location=device)
        finetuned_model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        logger.error(f"Error loading weights: {e}")
    
    # Ensure LoRA is enabled
    if hasattr(finetuned_model, 'enable_adapters'):
        finetuned_model.enable_adapters()
    finetuned_model.eval()
    
    # Load trial pairs
    with open(trial_file, 'r') as f:
        trial_pairs = f.readlines()[:max_trials]
    
    # Process trial pairs
    pretrained_scores = []
    finetuned_scores = []
    labels = []
    
    for line in trial_pairs:
        parts = line.strip().split()
        if len(parts) == 3:
            label = int(parts[0])
            file1 = os.path.join(audio_root, parts[1])
            file2 = os.path.join(audio_root, parts[2])
            
            # Load audios
            waveform1, sr1 = load_audio(file1)
            inputs1 = processor(
                waveform1.numpy(), 
                sampling_rate=sr1, 
                return_tensors="pt"
            ).input_values.to(device)
            
            waveform2, sr2 = load_audio(file2)
            inputs2 = processor(
                waveform2.numpy(), 
                sampling_rate=sr2, 
                return_tensors="pt"
            ).input_values.to(device)
            
            # Get embeddings from pretrained model
            with torch.no_grad():
                outputs1 = pretrained_model(inputs1)
                emb1_pre = outputs1.last_hidden_state.mean(dim=1)
                emb1_pre = torch.nn.functional.normalize(emb1_pre, p=2, dim=1)
                
                outputs2 = pretrained_model(inputs2)
                emb2_pre = outputs2.last_hidden_state.mean(dim=1)
                emb2_pre = torch.nn.functional.normalize(emb2_pre, p=2, dim=1)
                
                # Calculate pretrained similarity
                pre_sim = torch.sum(emb1_pre * emb2_pre, dim=1).item()
                pretrained_scores.append(pre_sim)
            
            # Get embeddings from finetuned model
            with torch.no_grad():
                outputs1 = finetuned_model.backbone(inputs1)
                emb1_ft = outputs1.last_hidden_state.mean(dim=1)
                emb1_ft = torch.nn.functional.normalize(emb1_ft, p=2, dim=1)
                
                outputs2 = finetuned_model.backbone(inputs2)
                emb2_ft = outputs2.last_hidden_state.mean(dim=1)
                emb2_ft = torch.nn.functional.normalize(emb2_ft, p=2, dim=1)
                
                # Calculate finetuned similarity
                ft_sim = torch.sum(emb1_ft * emb2_ft, dim=1).item()
                finetuned_scores.append(ft_sim)
            
            labels.append(label)
            
            # Log the pair and scores
            logger.info(f"Trial pair {len(pretrained_scores)}:")
            logger.info(f"  Label: {label}")
            logger.info(f"  Pretrained score: {pre_sim:.6f}")
            logger.info(f"  Finetuned score: {ft_sim:.6f}")
            logger.info(f"  Difference: {abs(pre_sim - ft_sim):.6f}")
    
    # Calculate average difference
    diffs = [abs(p - f) for p, f in zip(pretrained_scores, finetuned_scores)]
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    
    logger.info(f"Average score difference: {avg_diff:.6f}")
    
    # If average difference is very small, we have a problem
    if avg_diff < 1e-6:
        logger.error("CRITICAL: Scores are identical! LoRA weights are not being applied.")
        return False
    else:
        logger.info("Scores are different, suggesting LoRA is working to some extent.")
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose issues with finetuned model")
    parser.add_argument("--model", type=str, default="wavlm_base_plus", 
                       choices=["hubert_large", "wav2vec2_xlsr", "unispeech_sat", "wavlm_base_plus"],
                       help="Model name")
    parser.add_argument("--finetuned_model_path", type=str, required=True,
                       help="Path to finetuned model weights")
    parser.add_argument("--trial_file", type=str, required=True,
                       help="Path to trial file")
    parser.add_argument("--audio_root", type=str, required=True,
                       help="Root directory for audio files")
    parser.add_argument("--sample_file", type=str,
                       help="Sample audio file for embedding comparison (optional)")
    
    args = parser.parse_args()
    
    # Run diagnostics
    has_nonzero_lora = diagnose_model_weights(args.finetuned_model_path)
    
    embeddings_differ_dynamic, embeddings_differ_merged = False, False
    # If we have a sample file, compare embeddings
    if args.sample_file:
        embeddings_differ_dynamic, embeddings_differ_merged = compare_embeddings(args.model, args.finetuned_model_path, args.sample_file)
    else:
        # Use the first file from trial file as sample
        with open(args.trial_file, 'r') as f:
            first_line = f.readline().strip().split()
            if len(first_line) >= 2:
                sample_file = os.path.join(args.audio_root, first_line[1])
                embeddings_differ_dynamic, embeddings_differ_merged = compare_embeddings(args.model, args.finetuned_model_path, sample_file)
            else:
                logger.error("Could not find a sample file in the trial file")
                
    # Evaluate some trial pairs (keep this to check dynamic scores)
    scores_differ = evaluate_trial_pairs(args.model, args.finetuned_model_path, args.trial_file, args.audio_root, max_trials=5)
    
    # Final diagnosis based on merge results
    logger.info("--- FINAL DIAGNOSIS ---")
    if not has_nonzero_lora:
        logger.error("DIAGNOSIS: LoRA weights in saved model are all zero or very close to zero.")
        logger.error("You need to retrain with proper LoRA configuration and ensure weights are updated.")
    elif not embeddings_differ_merged:
        logger.error("DIAGNOSIS: LoRA weights were likely trained but are mathematically ineffective OR the merge process failed.")
        logger.error("Check training loss curve. If loss decreased, the issue might be subtle (e.g., numerical precision, merge bug). If loss didn't decrease, weights weren't trained effectively.")
    elif not embeddings_differ_dynamic:
        logger.error("DIAGNOSIS: LoRA weights ARE effective when merged, but PEFT is failing to apply them dynamically during inference.")
        logger.error("This points to an issue with PEFT's inference hooks, model structure compatibility, or eval/no_grad state interaction. Check PEFT/Transformers versions.")
    elif not scores_differ:
         logger.error("DIAGNOSIS: Embeddings differ but scores are identical. Issue might be in the scoring logic or the difference is too small to affect scores.")
    else:
        logger.info("DIAGNOSIS: LoRA seems to be applied dynamically and affects embeddings/scores.")
        logger.info("If performance isn't as expected, consider adjusting LoRA hyperparameters (r, alpha, learning rate) and retraining.")
        
    logger.info("-----------------------")