import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import h5py
import sys
from utils import load_subgraph_margin, get_item, get_item_edge

# Import necessary model classes to make them available for torch.load
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current directory to path
from TGN.tgn import TGN
from GraphM import GraphMixer
from models import *

def main(args):
    # Set device
    if args.device.lower() == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Set default degree based on dataset
    degree_dict = {"wikipedia": 20, "reddit": 20, "uci": 30, "mooc": 60, 
                  "enron": 30, "enron_sampled": 30, "canparl": 30, "uslegis": 30}
    if args.n_degree is None:
        args.n_degree = degree_dict.get(args.data, 20)
    print(f"Using n_degree={args.n_degree}")
    
    # Load the base model - with weights_only=False to bypass the security restriction
    model_path = os.path.join('params', 'tgnn', f'{args.base_type}_{args.data}.pt')
    print(f"Loading base model from {model_path}")
    try:
        # Try direct loading first
        base_model = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading model directly: {e}")
        print("Attempting with torch.serialization.add_safe_globals...")
        try:
            # Try with safe_globals
            from torch.serialization import safe_globals
            with safe_globals(["TGN", "GraphMixer", "TGAT", "TempME_Explainer"]):
                base_model = torch.load(model_path, map_location=device)
        except Exception as e2:
            print(f"Error with safe_globals: {e2}")
            # Last resort - try pickle
            import pickle
            with open(model_path, 'rb') as f:
                base_model = pickle.load(f)
                base_model = base_model.to(device)
    
    # Load the explainer
    explainer_path = os.path.join('params', 'explainer', args.base_type, f'{args.data}.pt')
    print(f"Loading explainer from {explainer_path}")
    try:
        explainer = torch.load(explainer_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading explainer: {e}")
        try:
            # Try with safe_globals
            from torch.serialization import safe_globals
            with safe_globals(["TGN", "GraphMixer", "TGAT", "TempME_Explainer"]):
                explainer = torch.load(explainer_path, map_location=device)
        except Exception as e2:
            print(f"Error with safe_globals for explainer: {e2}")
            # Last resort - try pickle
            import pickle
            with open(explainer_path, 'rb') as f:
                explainer = pickle.load(f)
                explainer = explainer.to(device)
    
    # Make sure models are on the correct device
    base_model = base_model.to(device)
    explainer = explainer.to(device)
    
    # Set models to evaluation mode
    base_model.eval()
    explainer.eval()
    
    # Load test data
    data_path = os.path.join('processed', f'{args.data}_test_cat.h5')
    print(f"Loading test data from {data_path}")
    pre_load_test = h5py.File(data_path, 'r')
    
    # Make sure args has all necessary attributes
    args.device = device
    
    test_pack = load_subgraph_margin(args, pre_load_test)
    edge_path = os.path.join('processed', f'{args.data}_test_edge.npy')
    test_edge = np.load(edge_path)
    
    # Get a few examples - ensure we don't try to get more than what's available
    max_samples = len(test_pack[0])  # Assuming first element contains all samples
    batch_size = min(5, max_samples)
    print(f"Visualizing {batch_size} examples")
    batch_idx = np.arange(batch_size)
    
    # Get subgraphs and walks
    print("Extracting subgraphs and walks")
    subgraph_src, subgraph_tgt, subgraph_bgd, walks_src, walks_tgt, walks_bgd, dst_l_fake = get_item(test_pack, batch_idx)
    src_edge, tgt_edge, bgd_edge = get_item_edge(test_edge, batch_idx)
    
    # Helper function to move data to device
    def move_to_device(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, tuple) or isinstance(data, list):
            return [move_to_device(x, device) for x in data]
        else:
            return data
    
    # Move all tensors to the same device
    print("Moving data to device...")
    walks_src = move_to_device(walks_src, device)
    walks_tgt = move_to_device(walks_tgt, device)
    walks_bgd = move_to_device(walks_bgd, device)
    src_edge = move_to_device(src_edge, device)
    tgt_edge = move_to_device(tgt_edge, device)
    bgd_edge = move_to_device(bgd_edge, device)
    
    # Generate timestamps (just an example, adjust to your dataset)
    ts_l_cut = torch.zeros(batch_size, device=device)
    
    # Ensure subgraph components are on the correct device
    subgraph_src = move_to_device(subgraph_src, device)
    subgraph_tgt = move_to_device(subgraph_tgt, device)
    subgraph_bgd = move_to_device(subgraph_bgd, device)
    
    # Move dst_l_fake to device if it's a tensor
    dst_l_fake = move_to_device(dst_l_fake, device)
    
    # Print information about input data
    print("\n===== INPUT DATA SUMMARY =====")
    print(f"Batch size: {batch_size}")
    
    # Print information about the walks (temporal graph walks)
    print("\nSource walks sample:")
    if isinstance(walks_src, list) and len(walks_src) > 0:
        print(f"  Walk structure: {len(walks_src)} components")
        for i, comp in enumerate(walks_src):
            if isinstance(comp, torch.Tensor):
                print(f"  Component {i}: shape {comp.shape}, dtype {comp.dtype}")
                if i == 0 and comp.numel() > 0:  # Print a small sample from first component
                    print(f"  Sample data: {comp[0][:5]}")
    
    # Print information about the edges
    print("\nSource edges sample:")
    if isinstance(src_edge, torch.Tensor):
        print(f"  Shape: {src_edge.shape}, dtype: {src_edge.dtype}")
        if src_edge.numel() > 0:
            print(f"  First few edges: {src_edge[0][:5]}")
    
    print("Generating explanations")
    try:
        with torch.no_grad():
            # Get explanations
            graphlet_imp_src = explainer(walks_src, ts_l_cut, src_edge)
            graphlet_imp_tgt = explainer(walks_tgt, ts_l_cut, tgt_edge)
            graphlet_imp_bgd = explainer(walks_bgd, ts_l_cut, bgd_edge)
            
            explanation = explainer.retrieve_explanation(
                subgraph_src, graphlet_imp_src, walks_src,
                subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                training=False
            )
    except Exception as e:
        print(f"Error generating explanations: {e}")
        print("Trying with CPU device...")
        
        # Fallback to CPU
        device = torch.device('cpu')
        base_model = base_model.to(device)
        explainer = explainer.to(device)
        
        # Move all data to CPU
        walks_src = move_to_device(walks_src, device)
        walks_tgt = move_to_device(walks_tgt, device)
        walks_bgd = move_to_device(walks_bgd, device)
        src_edge = move_to_device(src_edge, device)
        tgt_edge = move_to_device(tgt_edge, device)
        bgd_edge = move_to_device(bgd_edge, device)
        ts_l_cut = torch.zeros(batch_size, device=device)
        subgraph_src = move_to_device(subgraph_src, device)
        subgraph_tgt = move_to_device(subgraph_tgt, device)
        subgraph_bgd = move_to_device(subgraph_bgd, device)
        dst_l_fake = move_to_device(dst_l_fake, device)
        
        with torch.no_grad():
            # Get explanations
            graphlet_imp_src = explainer(walks_src, ts_l_cut, src_edge)
            graphlet_imp_tgt = explainer(walks_tgt, ts_l_cut, tgt_edge)
            graphlet_imp_bgd = explainer(walks_bgd, ts_l_cut, bgd_edge)
            
            explanation = explainer.retrieve_explanation(
                subgraph_src, graphlet_imp_src, walks_src,
                subgraph_tgt, graphlet_imp_tgt, walks_tgt,
                subgraph_bgd, graphlet_imp_bgd, walks_bgd,
                training=False
            )
    
    # Print output information
    print("\n===== OUTPUT SUMMARY =====")
    print(f"Explanation type: {type(explanation)}")
    
    if isinstance(explanation, tuple) or isinstance(explanation, list):
        print(f"Explanation components: {len(explanation)}")
        for i, item in enumerate(explanation):
            if isinstance(item, torch.Tensor):
                print(f"  Component {i}: shape {item.shape}, dtype {item.dtype}")
                if item.numel() > 0:
                    # Try to print the first few values of the first element
                    if len(item.shape) >= 2:
                        print(f"  Sample values: {item[0][:5]}")
                    else:
                        print(f"  Sample values: {item[:5]}")
    
    # For graphlet importance tensors:
    print("\nImportance matrices:")
    print(f"  Source importance: {graphlet_imp_src.shape}")
    print(f"  Target importance: {graphlet_imp_tgt.shape}")
    print(f"  Background importance: {graphlet_imp_bgd.shape}")
    
    # Sample values from importance matrices
    print("\nImportance matrix samples (first element, first 5 values):")
    if graphlet_imp_src.numel() > 0:
        if len(graphlet_imp_src.shape) >= 2:
            print(f"  Source: {graphlet_imp_src[0][0:5]}")
        else:
            print(f"  Source: {graphlet_imp_src[0:5]}")
    
    # Visualize
    vis_dir = 'explanation_visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Saving visualizations to {vis_dir}")
    
    # Process explanations and visualize them
    try:
        for i in range(batch_size):
            # Get edge importance for this sample
            edge_imp = explanation[0][i].cpu().numpy()
            
            # Print information about what we're visualizing
            print(f"\nVisualizing sample {i}:")
            print(f"  Edge importance shape: {edge_imp.shape}")
            print(f"  Edge importance range: min={edge_imp.min():.4f}, max={edge_imp.max():.4f}, mean={edge_imp.mean():.4f}")
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            plt.title(f"Edge Importance for Sample {i}")
            
            if len(edge_imp.shape) == 1:
                # Linear plot for 1D importance
                plt.bar(range(len(edge_imp)), edge_imp)
                plt.xlabel("Edge Index")
                plt.ylabel("Importance")
            else:
                # Heatmap for 2D importance
                sns.heatmap(edge_imp, cmap='viridis')
            
            plt.tight_layout()
            save_path = os.path.join(vis_dir, f"sample_{i}.png")
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved visualization for sample {i} to {save_path}")
            
        # Also create a visualization showing source, target, and background importances
        for sample_idx in range(min(2, batch_size)):  # Show first 2 samples
            plt.figure(figsize=(15, 5))
            
            print(f"\nVisualizing component importances for sample {sample_idx}:")
            
            src_imp = graphlet_imp_src[sample_idx].cpu().numpy()
            tgt_imp = graphlet_imp_tgt[sample_idx].cpu().numpy()
            bgd_imp = graphlet_imp_bgd[sample_idx].cpu().numpy()
            
            print(f"  Source importance: shape={src_imp.shape}, range=[{src_imp.min():.4f}, {src_imp.max():.4f}], mean={src_imp.mean():.4f}")
            print(f"  Target importance: shape={tgt_imp.shape}, range=[{tgt_imp.min():.4f}, {tgt_imp.max():.4f}], mean={tgt_imp.mean():.4f}")
            print(f"  Background importance: shape={bgd_imp.shape}, range=[{bgd_imp.min():.4f}, {bgd_imp.max():.4f}], mean={bgd_imp.mean():.4f}")
            
            # Source importance
            plt.subplot(1, 3, 1)
            plt.title(f"Source Importance (Sample {sample_idx})")
            plt.imshow(src_imp, cmap='viridis', aspect='auto')
            plt.colorbar()
            
            # Target importance
            plt.subplot(1, 3, 2)
            plt.title(f"Target Importance (Sample {sample_idx})")
            plt.imshow(tgt_imp, cmap='viridis', aspect='auto')
            plt.colorbar()
            
            # Background importance
            plt.subplot(1, 3, 3)
            plt.title(f"Background Importance (Sample {sample_idx})")
            plt.imshow(bgd_imp, cmap='viridis', aspect='auto')
            plt.colorbar()
            
            plt.tight_layout()
            save_path = os.path.join(vis_dir, f"components_sample_{sample_idx}.png")
            plt.savefig(save_path)
            plt.close()
            
            print(f"Saved component visualization for sample {sample_idx} to {save_path}")
        
        print(f"Saved all visualizations to '{vis_dir}' folder")
    except Exception as e:
        print(f"Error during visualization: {e}")
        # Print explanation structure to debug
        print("Explanation structure:")
        print(f"Type: {type(explanation)}")
        if isinstance(explanation, tuple) or isinstance(explanation, list):
            print(f"Length: {len(explanation)}")
            for i, item in enumerate(explanation):
                print(f"Item {i} type: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"Item {i} shape: {item.shape}")
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_type', type=str, default="tgn", help='tgn or graphmixer or tgat')
    parser.add_argument('--data', type=str, default="wikipedia", help='dataset name')
    parser.add_argument('--n_degree', type=int, default=None, help='number of neighbors to sample')
    parser.add_argument('--device', type=str, default="cuda", help='cpu or cuda')  # Changed default to CPU
    args = parser.parse_args()
    
    main(args)