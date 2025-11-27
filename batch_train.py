"""
Automated batch training script for TempME
Trains all models sequentially or in parallel (if multiple GPUs available)
"""

import subprocess
import sys
import os
import time
from datetime import datetime, timedelta

class TrainingPipeline:
    def __init__(self, dataset="enron_sampled", base_epochs=50, explainer_epochs=40):
        self.dataset = dataset
        self.base_epochs = base_epochs
        self.explainer_epochs = explainer_epochs
        self.models = ["tgn", "tgat", "graphmixer"]
        self.start_time = None
        self.results = {}
        
    def run_command(self, cmd, description):
        """Run a command and track its execution time"""
        print("\n" + "="*80)
        print(f"RUNNING: {description}")
        print("="*80)
        print(f"Command: {cmd}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*80)
        
        start = time.time()
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
            elapsed = time.time() - start
            
            print("-"*80)
            print(f"✓ COMPLETED in {elapsed/60:.1f} minutes")
            self.results[description] = {"status": "success", "time": elapsed}
            return True
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            print("-"*80)
            print(f"✗ FAILED after {elapsed/60:.1f} minutes")
            print(f"Error: {e}")
            self.results[description] = {"status": "failed", "time": elapsed}
            return False
    
    def train_all_sequential(self):
        """Train all models sequentially"""
        print("\n" + "="*80)
        print("TempME SEQUENTIAL TRAINING PIPELINE")
        print("="*80)
        print(f"Dataset: {self.dataset}")
        print(f"Models: {', '.join(self.models)}")
        print(f"Base epochs: {self.base_epochs}")
        print(f"Explainer epochs: {self.explainer_epochs}")
        print("="*80)
        
        self.start_time = time.time()
        
        for model in self.models:
            print(f"\n{'='*80}")
            print(f"TRAINING MODEL: {model.upper()}")
            print(f"{'='*80}")
            
            # Train base model
            base_cmd = f"python learn_base.py --base_type {model} --data {self.dataset} --n_epoch {self.base_epochs}"
            if not self.run_command(base_cmd, f"{model.upper()} - Base Model"):
                print(f"⚠ Warning: Base model training failed for {model}")
                continue
            
            # Train explainer
            exp_cmd = f"python temp_exp_main.py --base_type {model} --data {self.dataset} --n_epoch {self.explainer_epochs}"
            if not self.run_command(exp_cmd, f"{model.upper()} - Explainer"):
                print(f"⚠ Warning: Explainer training failed for {model}")
                continue
            
            # Verify enhancement
            verify_cmd = f"python enhance_main.py --data {self.dataset} --base_type {model}"
            self.run_command(verify_cmd, f"{model.upper()} - Verification")
        
        self.print_summary()
    
    def train_bases_then_explainers(self):
        """Train all base models first, then all explainers"""
        print("\n" + "="*80)
        print("TempME TWO-PHASE TRAINING PIPELINE")
        print("="*80)
        print(f"Dataset: {self.dataset}")
        print(f"Models: {', '.join(self.models)}")
        print(f"Base epochs: {self.base_epochs}")
        print(f"Explainer epochs: {self.explainer_epochs}")
        print("="*80)
        
        self.start_time = time.time()
        
        # Phase 1: Train all base models
        print("\n" + "="*80)
        print("PHASE 1: Training Base Models")
        print("="*80)
        
        for model in self.models:
            base_cmd = f"python learn_base.py --base_type {model} --data {self.dataset} --n_epoch {self.base_epochs}"
            self.run_command(base_cmd, f"{model.upper()} - Base Model")
        
        # Phase 2: Train all explainers
        print("\n" + "="*80)
        print("PHASE 2: Training Explainers")
        print("="*80)
        
        for model in self.models:
            exp_cmd = f"python temp_exp_main.py --base_type {model} --data {self.dataset} --n_epoch {self.explainer_epochs}"
            self.run_command(exp_cmd, f"{model.upper()} - Explainer")
        
        # Phase 3: Verify all
        print("\n" + "="*80)
        print("PHASE 3: Verification")
        print("="*80)
        
        for model in self.models:
            verify_cmd = f"python enhance_main.py --data {self.dataset} --base_type {model}"
            self.run_command(verify_cmd, f"{model.upper()} - Verification")
        
        self.print_summary()
    
    def train_single_model_full(self, model_type):
        """Train a single model with full pipeline"""
        print("\n" + "="*80)
        print(f"TempME SINGLE MODEL TRAINING: {model_type.upper()}")
        print("="*80)
        print(f"Dataset: {self.dataset}")
        print(f"Base epochs: {self.base_epochs}")
        print(f"Explainer epochs: {self.explainer_epochs}")
        print("="*80)
        
        self.start_time = time.time()
        
        # Train base model
        base_cmd = f"python learn_base.py --base_type {model_type} --data {self.dataset} --n_epoch {self.base_epochs}"
        if not self.run_command(base_cmd, f"{model_type.upper()} - Base Model"):
            print("✗ Training failed")
            return
        
        # Train explainer
        exp_cmd = f"python temp_exp_main.py --base_type {model_type} --data {self.dataset} --n_epoch {self.explainer_epochs}"
        if not self.run_command(exp_cmd, f"{model_type.upper()} - Explainer"):
            print("✗ Training failed")
            return
        
        # Verify
        verify_cmd = f"python enhance_main.py --data {self.dataset} --base_type {model_type}"
        self.run_command(verify_cmd, f"{model_type.upper()} - Verification")
        
        self.print_summary()
    
    def print_summary(self):
        """Print training summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        print(f"\nTotal execution time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        print(f"Started: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\nTask Results:")
        print("-"*80)
        
        success_count = 0
        total_task_time = 0
        
        for task, result in self.results.items():
            status_symbol = "✓" if result["status"] == "success" else "✗"
            task_time = result["time"] / 60
            total_task_time += result["time"]
            
            print(f"{status_symbol} {task:.<50} {task_time:>6.1f} min")
            
            if result["status"] == "success":
                success_count += 1
        
        print("-"*80)
        print(f"Tasks completed: {success_count}/{len(self.results)}")
        
        if success_count == len(self.results):
            print("\n✅ ALL TASKS COMPLETED SUCCESSFULLY!")
        else:
            print(f"\n⚠ {len(self.results) - success_count} task(s) failed")
        
        print("\nModel Parameters Saved:")
        print("-"*80)
        for model in self.models:
            base_path = f"params/tgnn/{model}_{self.dataset}.pt"
            exp_path = f"params/explainer/{model}/{self.dataset}.pt"
            
            if os.path.exists(base_path):
                print(f"✓ {base_path}")
            else:
                print(f"✗ {base_path} (not found)")
            
            if os.path.exists(exp_path):
                print(f"✓ {exp_path}")
            else:
                print(f"✗ {exp_path} (not found)")
        
        print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated TempME Training Pipeline')
    parser.add_argument('--mode', type=str, default='sequential',
                       choices=['sequential', 'two-phase', 'single'],
                       help='Training mode: sequential (model-by-model), two-phase (all bases then explainers), single (one model)')
    parser.add_argument('--dataset', type=str, default='enron_sampled',
                       help='Dataset to use (default: enron_sampled)')
    parser.add_argument('--base_epochs', type=int, default=50,
                       help='Number of epochs for base model training')
    parser.add_argument('--explainer_epochs', type=int, default=40,
                       help='Number of epochs for explainer training')
    parser.add_argument('--model', type=str, default='tgn',
                       choices=['tgn', 'tgat', 'graphmixer'],
                       help='Model type (only for single mode)')
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline(
        dataset=args.dataset,
        base_epochs=args.base_epochs,
        explainer_epochs=args.explainer_epochs
    )
    
    if args.mode == 'sequential':
        pipeline.train_all_sequential()
    elif args.mode == 'two-phase':
        pipeline.train_bases_then_explainers()
    elif args.mode == 'single':
        pipeline.train_single_model_full(args.model)


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    TempME AUTOMATED TRAINING PIPELINE                         ║
║                                                                               ║
║  This script automates the training of TGN, TGAT, and GraphMixer models      ║
║  along with their explainers for rapid experimentation.                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    main()
