import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import logging
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrained_eval import (
    load_pretrained_model, 
    VoxCelebTrialDataset, 
    custom_collate_fn,
    extract_embeddings,
    compute_similarity,
    calculate_eer,
    calculate_tar_at_far
)
from finetune import (
    SpeakerVerificationModel,
    apply_lora,
    get_embedding_dim,
    VoxCelebSpeakerDataset
)
from utils import setup_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
setup_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def extract_finetuned_embeddings(model, inputs):
    """
    Extract embeddings from a finetuned SpeakerVerificationModel with LoRA adapters.
    This handles both regular and PEFT model structures correctly.
    
    Args:
        model: The finetuned model
        inputs: Input tensor
        
    Returns:
        embeddings: Extracted embeddings
    """
    # Move inputs to device
    inputs = inputs.to(device)
    
    with torch.no_grad():
        # CRITICAL FIX: Access the backbone model directly, which should have the LoRA weights applied
        if hasattr(model, 'backbone'):
            # Use the backbone directly with LoRA adapters applied
            outputs = model.backbone(inputs)
            
            # Get embeddings from backbone outputs
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                # Try to handle different output structures
                logger.warning(f"Unexpected backbone output structure: {type(outputs)}")
                if isinstance(outputs, dict) and 'last_hidden_state' in outputs:
                    embeddings = outputs['last_hidden_state'].mean(dim=1)
                else:
                    # Last resort: try using the model's forward method
                    embeddings = model(inputs)
                    # If it returns a tuple (logits, embeddings), get the embeddings
                    if isinstance(embeddings, tuple) and len(embeddings) == 2:
                        embeddings = embeddings[1]
        else:
            # If model doesn't have a backbone attribute, use it directly
            embeddings = model(inputs)
            if isinstance(embeddings, tuple) and len(embeddings) == 2:
                embeddings = embeddings[1]
        
        # Normalize embeddings for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings

def extract_pretrained_embeddings(model, inputs):
    """
    Extract embeddings from a pretrained model.
    Wrapper around extract_embeddings with added logging.
    
    Args:
        model: The pretrained model
        inputs: Input tensor
        
    Returns:
        embeddings: Extracted embeddings
    """
    # Move inputs to device
    inputs = inputs.to(device)
    
    # Log input shape for debugging
    logger.info(f"Pretrained model input shape: {inputs.shape}")
    
    with torch.no_grad():
        # Use standard extraction from pretrained_eval
        embeddings = extract_embeddings(model, inputs)
        logger.info(f"Pretrained embeddings shape: {embeddings.shape}")
        
        return embeddings

def debug_compare_embeddings(pretrained_model, finetuned_model, processor, audio_path):
    """
    Debug function to compare embeddings from pretrained and finetuned models
    on the same input.
    
    Args:
        pretrained_model: The pretrained model
        finetuned_model: The finetuned model
        processor: Audio processor
        audio_path: Path to a sample audio file
    """
    logger.info(f"Comparing embeddings on sample audio: {audio_path}")
    
    # Process audio
    try:
        # Load audio using processor
        audio_dict = processor(audio_path, sampling_rate=16000, return_tensors="pt")
        audio_input = audio_dict.input_values
        
        # Move to device
        audio_input = audio_input.to(device)
        
        # Get pretrained embedding
        pretrained_model.eval()
        pretrained_embedding = extract_pretrained_embeddings(pretrained_model, audio_input)
        
        # Get finetuned embedding
        finetuned_model.eval()
        finetuned_embedding = extract_finetuned_embeddings(finetuned_model, audio_input)
        
        # Compare embeddings
        diff = torch.norm(pretrained_embedding - finetuned_embedding).item()
        logger.info(f"Embedding difference (L2 norm): {diff}")
        
        # Show sample of weights before and after
        if hasattr(finetuned_model, 'backbone'):
            # Get a parameter from backbone to compare
            for name, param in finetuned_model.backbone.named_parameters():
                if param.requires_grad:
                    logger.info(f"Sample of finetuned weight '{name}': {param.data.flatten()[:5]}")
                    # Try to find the same parameter in pretrained model
                    for pretrained_name, pretrained_param in pretrained_model.named_parameters():
                        if pretrained_name == name:
                            logger.info(f"Sample of pretrained weight '{pretrained_name}': {pretrained_param.data.flatten()[:5]}")
                            logger.info(f"Weight difference: {torch.norm(param.data - pretrained_param.data).item()}")
                            break
                    break
        
        return pretrained_embedding, finetuned_embedding
        
    except Exception as e:
        logger.error(f"Error in debug_compare_embeddings: {e}")
        return None, None

def verify_weight_loading(model, saved_weights_path):
    """
    Verify that weights are properly loaded by comparing a few parameters
    before and after loading.
    """
    # First, try to fix the naming mismatch between saved and model weights
    logger.info(f"Loading weights from {saved_weights_path}")
    
    try:
        # Load the state dict from file
        saved_state_dict = torch.load(saved_weights_path, map_location=device)
        
        # DIAGNOSTIC: Print out keys in the saved model that contain 'lora'
        logger.info("Keys in saved model containing 'lora':")
        lora_keys_in_saved = [k for k in saved_state_dict.keys() if 'lora' in k.lower()]
        for idx, key in enumerate(lora_keys_in_saved[:10]):  # Show up to 10 keys
            logger.info(f"  {idx}: {key}: shape={saved_state_dict[key].shape}, sum={saved_state_dict[key].sum().item()}")
        
        if not lora_keys_in_saved:
            logger.error("CRITICAL: No LoRA keys found in saved weights! The model was probably not saved with LoRA layers.")
            
        # DIAGNOSTIC: Print model keys containing 'lora'
        model_state_dict = model.state_dict()
        logger.info("Keys in model containing 'lora':")
        lora_keys_in_model = [k for k in model_state_dict.keys() if 'lora' in k.lower()]
        for idx, key in enumerate(lora_keys_in_model[:10]):  # Show up to 10 keys
            logger.info(f"  {idx}: {key}: shape={model_state_dict[key].shape}")
        
        if not lora_keys_in_model:
            logger.error("CRITICAL: No LoRA keys found in model! The apply_lora function might not be working correctly.")
            
        # Rest of your existing code for trying different key matching strategies...
        # [Keep your existing code here]
        
        # DIAGNOSTIC: Check if any LoRA parameters have non-zero values in the saved weights
        has_nonzero_lora = False
        for k in lora_keys_in_saved:
            param = saved_state_dict[k]
            if torch.abs(param).sum() > 1e-6:
                logger.info(f"Found non-zero LoRA parameter in saved weights: {k}")
                logger.info(f"  Sample values: {param.flatten()[:5]}")
                logger.info(f"  Norm: {torch.norm(param).item()}")
                has_nonzero_lora = True
                break
                
        if not has_nonzero_lora:
            logger.error("CRITICAL: All LoRA parameters in saved weights are zero or very small!")
            logger.error("This suggests the model wasn't properly trained or the LoRA weights weren't updated.")
            
        return has_nonzero_lora
            
    except Exception as e:
        logger.error(f"Error during weight loading: {e}")
        return False

def examine_saved_model(model_path):
    """
    Examine the saved model state dict to diagnose issues with LoRA parameters.
    
    Args:
        model_path: Path to the saved model state dict
    """
    logger.info(f"Examining saved model at {model_path}")
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Print basic stats
        logger.info(f"State dict contains {len(state_dict)} keys")
        
        # Check for LoRA parameters
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
        adapter_keys = [k for k in state_dict.keys() if 'adapter' in k.lower()]
        
        logger.info(f"Found {len(lora_keys)} keys containing 'lora'")
        logger.info(f"Found {len(adapter_keys)} keys containing 'adapter'")
        
        # If no LoRA keys found, look for any trainable parameters
        if not lora_keys and not adapter_keys:
            logger.warning("No LoRA or adapter keys found! Looking for any parameters that might be trainable...")
            
            # Print a few sample keys to see what's in the state dict
            sample_keys = list(state_dict.keys())[:10]
            logger.info(f"Sample keys in saved model: {sample_keys}")
            
            # Try to identify what kind of model was saved
            if any('backbone' in k for k in state_dict.keys()):
                logger.info("Model appears to have a 'backbone' structure (from SpeakerVerificationModel)")
            elif any('base_model' in k for k in state_dict.keys()):
                logger.info("Model appears to have a 'base_model' structure (from PEFT)")
            else:
                logger.info("Model appears to be a standard model without special structure")
        
        # If LoRA keys found, examine them
        nonzero_params = 0
        zero_params = 0
        
        for key_list, key_type in [(lora_keys, "LoRA"), (adapter_keys, "Adapter")]:
            for k in key_list:
                param = state_dict[k]
                param_sum = torch.abs(param).sum().item()
                
                if param_sum > 1e-6:
                    nonzero_params += 1
                    logger.info(f"Non-zero {key_type} parameter: {k}, shape={param.shape}, sum={param_sum:.6f}")
                    logger.info(f"  Sample values: {param.flatten()[:5]}")
                else:
                    zero_params += 1
        
        logger.info(f"Summary: Found {nonzero_params} non-zero and {zero_params} zero/near-zero LoRA/Adapter parameters")
        
        # Check if the model contains ArcFace weights
        arcface_keys = [k for k in state_dict.keys() if 'arcface' in k.lower()]
        if arcface_keys:
            logger.info(f"Found {len(arcface_keys)} ArcFace keys, suggesting this is a complete SpeakerVerificationModel")
            # Sample a few ArcFace parameters
            for k in arcface_keys[:3]:
                param = state_dict[k]
                logger.info(f"ArcFace parameter: {k}, shape={param.shape}, sum={torch.abs(param).sum().item():.6f}")
        else:
            logger.warning("No ArcFace parameters found. This might not be a complete SpeakerVerificationModel.")
            
        return nonzero_params > 0
        
    except Exception as e:
        logger.error(f"Error examining saved model: {e}")
        return False

def evaluate_model(model, processor, trial_file, audio_root, batch_size=4, model_name="", is_finetuned=False, debug_sample=None):
    """
    Evaluate model performance on speaker verification using VoxCeleb1 trial pairs.
    
    Args:
        model: The model to evaluate
        processor: The processor for the model
        trial_file: Path to the trial file
        audio_root: Root directory for audio files
        batch_size: Batch size for evaluation
        model_name: Name of the model (for logging)
        is_finetuned: Whether this is a finetuned model
        debug_sample: Optional debug sample to use for comparing embeddings
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name} on {trial_file}")
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Log model type
    logger.info(f"Model type: {'Finetuned' if is_finetuned else 'Pretrained'}")
    
    # Create dataset and dataloader
    dataset = VoxCelebTrialDataset(trial_file, audio_root, processor)
    logger.info(f"Created dataset with {len(dataset)} trial pairs")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # If debug_sample is provided, run a debug comparison
    if debug_sample is not None and 'pretrained_model' in debug_sample and 'finetuned_model' in debug_sample:
        # Get the first audio file from the dataset for debugging
        first_item = dataset[0]
        audio_path = first_item['audio1_path']
        debug_compare_embeddings(
            debug_sample['pretrained_model'], 
            debug_sample['finetuned_model'], 
            processor, 
            audio_path
        )
    
    # Collect all labels and scores
    all_labels = []
    all_scores = []
    
    # Store some sample embeddings for comparison
    sample_embeddings = []
    
    logger.info("Processing trial pairs...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
        labels = batch['label']
        audio1_samples = batch['audio1']
        audio2_samples = batch['audio2']
        
        # Process each sample individually
        batch_scores = []
        for i in range(len(audio1_samples)):
            # Extract embeddings for each pair
            with torch.no_grad():
                # Process first audio
                audio1 = audio1_samples[i].unsqueeze(0).to(device)
                if is_finetuned:
                    emb1 = extract_finetuned_embeddings(model, audio1)
                else:
                    emb1 = extract_pretrained_embeddings(model, audio1)
                
                # Process second audio
                audio2 = audio2_samples[i].unsqueeze(0).to(device)
                if is_finetuned:
                    emb2 = extract_finetuned_embeddings(model, audio2)
                else:
                    emb2 = extract_pretrained_embeddings(model, audio2)
                
                # Store some sample embeddings (just first batch)
                if batch_idx == 0 and i == 0:
                    sample_embeddings.append((model_name, emb1.cpu().numpy()))
                
                # Compute similarity
                score = compute_similarity(emb1, emb2).item()
                batch_scores.append(score)
        
        # Collect labels and scores
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(batch_scores)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    tar_at_far, far_threshold = calculate_tar_at_far(all_labels, all_scores, far_target=0.01)
    
    # Calculate verification accuracy (using optimal threshold)
    pred_labels = (all_scores >= eer_threshold).astype(int)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(all_labels, pred_labels)
    
    results = {
        'model_name': model_name,
        'eer': eer * 100,  # Convert to percentage
        'tar_at_1far': tar_at_far * 100,  # Convert to percentage
        'accuracy': accuracy * 100,  # Convert to percentage
        'eer_threshold': eer_threshold,
        'far_threshold': far_threshold,
        'num_trials': len(all_labels),
        'sample_embeddings': sample_embeddings
    }
    
    logger.info(f"Evaluation results for {model_name} on {len(all_labels)} trials:")
    logger.info(f"  EER: {eer*100:.2f}%")
    logger.info(f"  TAR@1%FAR: {tar_at_far*100:.2f}%")
    logger.info(f"  Accuracy: {accuracy*100:.2f}%")
    
    return results

def diagnostic_compare_models(pretrained_model, finetuned_model, processor, audio_file):
    """
    Compare embeddings from pretrained and finetuned models directly.
    
    Args:
        pretrained_model: The pretrained model
        finetuned_model: The finetuned model
        processor: The processor for audio
        audio_file: Path to an audio file for testing
    """
    # Load and process audio
    waveform, sr = load_audio(audio_file)
    inputs = processor(
        waveform.numpy(), 
        sampling_rate=sr, 
        return_tensors="pt"
    ).input_values.to(device)
    
    # Get embeddings from pretrained model
    pretrained_model.eval()
    with torch.no_grad():
        pretrained_outputs = pretrained_model(inputs)
        pretrained_emb = pretrained_outputs.last_hidden_state.mean(dim=1)
        pretrained_emb = torch.nn.functional.normalize(pretrained_emb, p=2, dim=1)
    
    # Get embeddings from finetuned model using LoRA
    finetuned_model.eval()
    with torch.no_grad():
        # Force using the backbone with LoRA adapters
        backbone_outputs = finetuned_model.backbone(inputs)
        finetuned_emb = backbone_outputs.last_hidden_state.mean(dim=1)
        finetuned_emb = torch.nn.functional.normalize(finetuned_emb, p=2, dim=1)
    
    # Compare embeddings
    diff = torch.norm(pretrained_emb - finetuned_emb).item()
    cos_sim = torch.sum(pretrained_emb * finetuned_emb, dim=1).item()
    
    logger.info(f"Embedding L2 difference: {diff}")
    logger.info(f"Embedding cosine similarity: {cos_sim}")
    
    return diff, cos_sim

def evaluate_with_custom_loop(model, processor, trial_file, audio_root, 
                              is_finetuned=False, batch_size=4, max_trials=4000):
    """
    Evaluate model with a custom loop that correctly handles the finetuned model.
    
    Args:
        model: The model to evaluate
        processor: The processor for audio
        trial_file: Path to trial file
        audio_root: Root directory for audio files
        is_finetuned: Whether this is a finetuned model
        batch_size: Batch size for evaluation
        max_trials: Maximum number of trials to evaluate
    
    Returns:
        dict: Dictionary with evaluation results
    """
    # Load trial pairs
    with open(trial_file, 'r') as f:
        trial_pairs = f.readlines()[:max_trials]
    
    # Parse trial pairs
    trials = []
    for line in trial_pairs:
        parts = line.strip().split()
        if len(parts) == 3:
            label = int(parts[0])
            file1 = os.path.join(audio_root, parts[1])
            file2 = os.path.join(audio_root, parts[2])
            trials.append((label, file1, file2))
    
    # Set model to eval mode
    model.eval()
    
    # Process trial pairs
    all_labels = []
    all_scores = []
    
    for i in range(0, len(trials), batch_size):
        batch_trials = trials[i:i+batch_size]
        batch_labels = []
        batch_scores = []
        
        for label, file1, file2 in batch_trials:
            # Load and process audio
            waveform1, sr1 = load_audio(file1)
            inputs1 = processor(waveform1.numpy(), sampling_rate=sr1, return_tensors="pt").input_values.to(device)
            
            waveform2, sr2 = load_audio(file2)
            inputs2 = processor(waveform2.numpy(), sampling_rate=sr2, return_tensors="pt").input_values.to(device)
            
            # Extract embeddings
            with torch.no_grad():
                if is_finetuned:
                    # Use backbone with LoRA for finetuned model
                    outputs1 = model.backbone(inputs1)
                    emb1 = outputs1.last_hidden_state.mean(dim=1)
                    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
                    
                    outputs2 = model.backbone(inputs2)
                    emb2 = outputs2.last_hidden_state.mean(dim=1)
                    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
                else:
                    # Use standard extraction for pretrained model
                    outputs1 = model(inputs1)
                    emb1 = outputs1.last_hidden_state.mean(dim=1)
                    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
                    
                    outputs2 = model(inputs2)
                    emb2 = outputs2.last_hidden_state.mean(dim=1)
                    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
                
                # Compute cosine similarity
                score = torch.sum(emb1 * emb2, dim=1).item()
            
            batch_labels.append(label)
            batch_scores.append(score)
        
        all_labels.extend(batch_labels)
        all_scores.extend(batch_scores)
        
        # Log progress
        if (i // batch_size) % 10 == 0:
            logger.info(f"Processed {i+len(batch_trials)}/{len(trials)} trial pairs")
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate metrics
    eer, eer_threshold = calculate_eer(all_labels, all_scores)
    tar_at_far, far_threshold = calculate_tar_at_far(all_labels, all_scores, far_target=0.01)
    
    # Calculate accuracy
    pred_labels = (all_scores >= eer_threshold).astype(int)
    accuracy = accuracy_score(all_labels, pred_labels)
    
    results = {
        'eer': eer * 100,
        'tar_at_1far': tar_at_far * 100,
        'accuracy': accuracy * 100,
        'num_trials': len(all_labels)
    }
    
    logger.info(f"Results with custom evaluation loop:")
    logger.info(f"  EER: {results['eer']:.2f}%")
    logger.info(f"  TAR@1%FAR: {results['tar_at_1far']:.2f}%")
    logger.info(f"  Accuracy: {results['accuracy']:.2f}%")
    
    return results

def compare_pretrained_and_finetuned(model_name, trial_file, audio_root, 
                                     finetuned_model_path, train_metadata, 
                                     output_dir, batch_size=4, pretrained_results_file=None,
                                     use_cached_pretrained=False, saved_subset_path=None):
    """
    Compare pretrained and finetuned models on speaker verification
    
    Args:
        model_name: Name of the model
        trial_file: Path to trial file
        audio_root: Root directory for audio files
        finetuned_model_path: Path to finetuned model weights
        train_metadata: Path to training metadata (for speaker count)
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        pretrained_results_file: Path to CSV with pretrained model results (if provided, skip pretrained eval)
        use_cached_pretrained: If True, use cached pretrained results instead of re-evaluating
        saved_subset_path: Path to saved subset of trial pairs (speeds up evaluation)
        
    Returns:
        tuple: (pretrained_results, finetuned_results)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DIAGNOSTIC: First examine the saved model to understand its structure
    logger.info("DIAGNOSTIC: Examining saved model structure")
    has_nonzero_lora = examine_saved_model(finetuned_model_path)
    
    if not has_nonzero_lora:
        logger.error("CRITICAL ISSUE: The saved model does not contain non-zero LoRA parameters!")
        logger.error("This suggests the model wasn't properly trained or LoRA weights weren't saved.")
        logger.error("You may need to retrain the model with proper LoRA configuration and weight saving.")
        # We'll continue anyway for diagnostic purposes
    
    # First check if we should use saved pretrained results
    if pretrained_results_file and use_cached_pretrained:
        logger.info(f"Loading pretrained results from {pretrained_results_file}")
        try:
            pretrained_df = pd.read_csv(pretrained_results_file)
            # Convert dataframe row to dictionary
            pretrained_results = pretrained_df.iloc[0].to_dict()
            logger.info(f"Loaded pretrained results: EER={pretrained_results.get('eer', 'N/A')}%")
        except Exception as e:
            logger.error(f"Error loading pretrained results: {e}")
            logger.info("Falling back to evaluating pretrained model")
            use_cached_pretrained = False
    else:
        use_cached_pretrained = False
    
    # If not using cached results, evaluate the pretrained model
    if not use_cached_pretrained:
        # Load the pretrained model
        logger.info(f"Loading pretrained model architecture for {model_name}")
        pretrained_model, processor = load_pretrained_model(model_name)
        pretrained_model.eval()
        
        # Log number of parameters in pretrained model
        pretrained_params = sum(p.numel() for p in pretrained_model.parameters())
        logger.info(f"Pretrained model has {pretrained_params:,} parameters")
        
        # If a subset path is provided, use it for faster evaluation
        effective_trial_file = saved_subset_path if saved_subset_path and os.path.exists(saved_subset_path) else trial_file
        if saved_subset_path and os.path.exists(saved_subset_path):
            logger.info(f"Using saved subset for faster evaluation: {saved_subset_path}")
        
        # Evaluate pretrained model
        logger.info("Evaluating pretrained model...")
        pretrained_results = evaluate_model(
            pretrained_model, 
            processor, 
            effective_trial_file, 
            audio_root, 
            batch_size, 
            f"Pretrained {model_name}",
            is_finetuned=False
        )
    
    # Now load the pretrained model architecture for finetuning
    logger.info(f"Loading pretrained model architecture for {model_name}")
    pretrained_model, processor = load_pretrained_model(model_name)
    pretrained_model.eval()
    
    # DIAGNOSTIC: Try direct forward pass with pretrained model on a sample
    logger.info("DIAGNOSTIC: Testing pretrained model forward pass")
    with torch.no_grad():
        # Create a dummy input of appropriate shape
        dummy_input = torch.randn(1, 16000).to(device)  # 1 second of audio at 16kHz
        # Get embeddings
        pretrained_outputs = pretrained_model(dummy_input)
        if hasattr(pretrained_outputs, 'last_hidden_state'):
            logger.info(f"Pretrained model outputs last_hidden_state with shape: {pretrained_outputs.last_hidden_state.shape}")
        else:
            logger.info(f"Pretrained model output type: {type(pretrained_outputs)}")
    
    # Now let's focus on evaluating the finetuned model
    logger.info(f"Setting up finetuned model architecture for {model_name}")
    
    # Get embedding dimension
    embedding_dim = get_embedding_dim(model_name)
    
    # Create model with same architecture as during training
    logger.info("Creating SpeakerVerificationModel with backbone and LoRA")
    finetuned_model = SpeakerVerificationModel(
        pretrained_model, 
        embedding_dim,
        100  # Use exactly 100 speakers as in the training
    ).to(device)
    
    # DIAGNOSTIC: Check model structure before applying LoRA
    logger.info("DIAGNOSTIC: Model structure before applying LoRA")
    logger.info(f"Model has backbone: {hasattr(finetuned_model, 'backbone')}")
    logger.info(f"Model has arcface: {hasattr(finetuned_model, 'arcface')}")
    
    # Apply LoRA with same configuration as during training
    logger.info("Applying LoRA configuration to model")
    finetuned_model = apply_lora(finetuned_model, model_name)
    
    # DIAGNOSTIC: Check model structure after applying LoRA
    logger.info("DIAGNOSTIC: Model structure after applying LoRA")
    logger.info(f"Model has base_model: {hasattr(finetuned_model, 'base_model')}")
    logger.info(f"Model has active_adapter: {hasattr(finetuned_model, 'active_adapter')}")
    
    # DIAGNOSTIC: Check for LoRA parameters
    lora_param_count = 0
    for name, param in finetuned_model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_param_count += 1
            if lora_param_count <= 5:  # Just show the first 5
                logger.info(f"LoRA parameter found: {name}, shape={param.shape}")
    logger.info(f"Total LoRA parameters found: {lora_param_count}")
    
    # DIAGNOSTIC: Ensure LoRA adapters are enabled
    if hasattr(finetuned_model, 'enable_adapters'):
        logger.info("Enabling LoRA adapters explicitly")
        finetuned_model.enable_adapters()
    elif hasattr(finetuned_model, 'base_model') and hasattr(finetuned_model.base_model, 'enable_adapters'):
        logger.info("Enabling LoRA adapters on base_model")
        finetuned_model.base_model.enable_adapters()
    else:
        logger.warning("No enable_adapters method found - adapters should be enabled by default")
    
    # Verify weight loading using our enhanced function
    logger.info(f"Verifying weight loading from {finetuned_model_path}")
    weights_changed = verify_weight_loading(finetuned_model, finetuned_model_path)
    if not weights_changed:
        logger.error("WARNING: Weights did not change after loading! This is a critical issue.")
        
        # DIAGNOSTIC: Try to directly inspect if the weights file contains LoRA params
        logger.info("DIAGNOSTIC: Checking if saved weights contain LoRA parameters")
        saved_weights = torch.load(finetuned_model_path, map_location=device)
        lora_keys = [k for k in saved_weights.keys() if 'lora' in k.lower()]
        logger.info(f"Found {len(lora_keys)} keys containing 'lora' in saved weights")
        if len(lora_keys) > 0:
            logger.info(f"Sample LoRA keys in saved weights: {lora_keys[:5]}")
            
            # DIAGNOSTIC: Check if these LoRA weights are actually non-zero
            for k in lora_keys[:5]:
                param = saved_weights[k]
                logger.info(f"LoRA param {k}: sum={torch.abs(param).sum().item()}, norm={torch.norm(param).item()}")
                if torch.abs(param).sum().item() < 1e-6:
                    logger.warning(f"LoRA parameter {k} is effectively zero!")
    else:
        logger.info("Weights successfully changed after loading.")
    
    # DIAGNOSTIC: Try direct forward pass with finetuned model on a sample
    logger.info("DIAGNOSTIC: Testing finetuned model forward pass")
    with torch.no_grad():
        # Create a dummy input of appropriate shape
        dummy_input = torch.randn(1, 16000).to(device)  # 1 second of audio at 16kHz
        
        # Try to use the model's forward method
        try:
            # First try with backbone
            if hasattr(finetuned_model, 'backbone'):
                backbone_outputs = finetuned_model.backbone(dummy_input)
                if hasattr(backbone_outputs, 'last_hidden_state'):
                    logger.info(f"Finetuned backbone outputs last_hidden_state with shape: {backbone_outputs.last_hidden_state.shape}")
                else:
                    logger.info(f"Finetuned backbone output type: {type(backbone_outputs)}")
            
            # Then try with full model
            full_outputs = finetuned_model(dummy_input)
            if isinstance(full_outputs, tuple) and len(full_outputs) == 2:
                logger.info(f"Finetuned model returns tuple of length 2 (likely logits and embeddings)")
            else:
                logger.info(f"Finetuned model output type: {type(full_outputs)}")
        except Exception as e:
            logger.error(f"Error during finetuned model forward pass: {e}")
    
    # Create debug info for comparing embeddings
    debug_sample = {
        'pretrained_model': pretrained_model if not use_cached_pretrained else None,
        'finetuned_model': finetuned_model
    }
    
    # If a subset path is provided, use it for faster evaluation
    effective_trial_file = saved_subset_path if saved_subset_path and os.path.exists(saved_subset_path) else trial_file
    if saved_subset_path and os.path.exists(saved_subset_path):
        logger.info(f"Using saved subset for faster evaluation: {saved_subset_path}")
    
    # Evaluate finetuned model
    logger.info("Evaluating finetuned model performance")
    finetuned_results = evaluate_model(
        finetuned_model, 
        processor, 
        effective_trial_file, 
        audio_root, 
        batch_size,
        f"Finetuned {model_name}",
        is_finetuned=True,
        debug_sample=debug_sample if not use_cached_pretrained else None
    )
    
    # Compare sample embeddings from each model
    if 'sample_embeddings' in pretrained_results and 'sample_embeddings' in finetuned_results:
        pretrained_emb = None
        finetuned_emb = None
        
        for model_name, emb in pretrained_results['sample_embeddings']:
            if model_name.startswith("Pretrained"):
                pretrained_emb = emb
                
        for model_name, emb in finetuned_results['sample_embeddings']:
            if model_name.startswith("Finetuned"):
                finetuned_emb = emb
        
        if pretrained_emb is not None and finetuned_emb is not None:
            # Calculate L2 distance between embeddings
            emb_diff = np.linalg.norm(pretrained_emb - finetuned_emb)
            logger.info(f"L2 distance between sample pretrained and finetuned embeddings: {emb_diff}")
            
            if emb_diff < 1e-6:
                logger.error("CRITICAL: Sample embeddings are identical between pretrained and finetuned models!")
            else:
                logger.info("Sample embeddings are different between pretrained and finetuned models.")
    
    # Verify the results are different
    eer_diff = abs(pretrained_results['eer'] - finetuned_results['eer'])
    if eer_diff < 1e-6:
        logger.error("CRITICAL: Pretrained and finetuned models have identical EER values!")
        logger.error("This suggests there might be an issue with the evaluation code or model loading.")
    else:
        logger.info(f"EER difference between pretrained and finetuned: {eer_diff:.4f}%")
    
    # Save results
    results_df = pd.DataFrame([pretrained_results, finetuned_results])
    # Remove sample_embeddings column before saving to CSV
    if 'sample_embeddings' in results_df.columns:
        results_df = results_df.drop('sample_embeddings', axis=1)
    results_file = output_dir / "comparison_results.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    # Plot comparison
    plot_comparison(pretrained_results, finetuned_results, model_name, output_dir)
    
    # Get a random audio sample for debugging
    sample_item = dataset[0]  # First item from your evaluation dataset
    sample_audio = sample_item['audio1'].unsqueeze(0).to(device)

    # Get embeddings from both models
    pretrained_emb = extract_pretrained_embeddings(pretrained_model, sample_audio)
    finetuned_emb = extract_finetuned_embeddings(finetuned_model, sample_audio)

    # Check if embeddings are nearly identical
    emb_diff = torch.norm(pretrained_emb - finetuned_emb).item()
    logger.info(f"Embedding difference between pretrained and finetuned: {emb_diff}")
    if emb_diff < 1e-6:
        logger.error("CRITICAL: Embeddings are identical between models!")
    
    # DIAGNOSTIC: Running direct model comparison on a sample file
    logger.info("DIAGNOSTIC: Running direct model comparison on a sample file")
    # Get first file from dataset for testing
    sample_item = dataset[0]
    audio_path = sample_item['audio1_path']
    diagnostic_compare_models(pretrained_model, finetuned_model, processor, audio_path)

    # Then use the custom evaluation loop
    logger.info("DIAGNOSTIC: Running custom evaluation loop that bypasses pretrained_eval")
    finetuned_results_custom = evaluate_with_custom_loop(
        finetuned_model,
        processor,
        effective_trial_file,
        audio_root,
        is_finetuned=True,
        batch_size=batch_size,
        max_trials=4000
    )

    logger.info("Comparison of standard vs custom evaluation:")
    logger.info(f"Standard EER: {finetuned_results['eer']:.2f}% | Custom EER: {finetuned_results_custom['eer']:.2f}%")
    
    return pretrained_results, finetuned_results

def plot_comparison(pretrained_results, finetuned_results, model_name, output_dir):
    """
    Plot comparison of pretrained and finetuned model results
    
    Args:
        pretrained_results: Dictionary with pretrained model results
        finetuned_results: Dictionary with finetuned model results
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    labels = [f"Pretrained {model_name}", f"Finetuned {model_name}"]
    metrics = ['eer', 'tar_at_1far', 'accuracy']
    metric_names = ['EER (%)', 'TAR@1%FAR (%)', 'Verification Accuracy (%)']
    
    plt.figure(figsize=(15, 5))
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(1, 3, i+1)
        
        # Check if the metric exists in both result sets
        if metric in pretrained_results and metric in finetuned_results:
            values = [pretrained_results[metric], finetuned_results[metric]]
            
            # For EER, lower is better
            if metric == 'eer':
                colors = ['red', 'green'] if values[0] > values[1] else ['green', 'red']
            else:  # For TAR and Accuracy, higher is better
                colors = ['red', 'green'] if values[0] < values[1] else ['green', 'red']
                
            plt.bar(labels, values, color=colors)
            plt.title(name)
            plt.ylabel('Percentage (%)')
            plt.xticks(rotation=15)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                plt.text(j, v + 1, f"{v:.2f}%", ha='center')
                
            # Add difference label
            diff = abs(values[1] - values[0])
            if metric == 'eer':
                # For EER, show improvement as negative (reduction in error)
                diff_label = f"-{diff:.2f}%" if values[1] < values[0] else f"+{diff:.2f}%"
            else:
                # For other metrics, show improvement as positive
                diff_label = f"+{diff:.2f}%" if values[1] > values[0] else f"-{diff:.2f}%"
            
            plt.text(0.5, max(values) + 5, f"Diff: {diff_label}", ha='center')
        else:
            # Handle the case where a metric is missing
            plt.title(f"{name} (Missing Data)")
            plt.ylabel('Percentage (%)')
            plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_plot.png")
    logger.info(f"Comparison plot saved to {output_dir / 'comparison_plot.png'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare speaker verification models")
    parser.add_argument("--model", type=str, default="wavlm_base_plus", 
                        choices=["hubert_large", "wav2vec2_xlsr", "unispeech_sat", "wavlm_base_plus"],
                        help="Pretrained model to evaluate")
    parser.add_argument("--trial_file", type=str, required=True,
                        help="Path to trial file for evaluation (VoxCeleb1)")
    parser.add_argument("--vox1_audio_root", type=str, required=True,
                        help="Root directory for VoxCeleb1 audio files")
    parser.add_argument("--train_metadata", type=str, required=True,
                        help="Path to training metadata file (for speaker count)")
    parser.add_argument("--finetuned_model_path", type=str, required=True,
                        help="Path to finetuned model weights")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    # Add new parameters
    parser.add_argument("--pretrained_results_file", type=str,
                        help="Path to CSV with pretrained model results (if provided, can skip pretrained eval)")
    parser.add_argument("--use_cached_pretrained", action="store_true",
                        help="Use cached pretrained results instead of re-evaluating")
    parser.add_argument("--saved_subset_path", type=str,
                        help="Path to saved subset of trial pairs (speeds up evaluation)")
    
    args = parser.parse_args()
    
    # Compare pretrained and finetuned models
    compare_pretrained_and_finetuned(
        model_name=args.model,
        trial_file=args.trial_file,
        audio_root=args.vox1_audio_root,
        finetuned_model_path=args.finetuned_model_path,
        train_metadata=args.train_metadata,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        pretrained_results_file=args.pretrained_results_file,
        use_cached_pretrained=args.use_cached_pretrained,
        saved_subset_path=args.saved_subset_path
    )


if __name__ == "__main__":
    main()