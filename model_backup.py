#!/usr/bin/env python3
"""
Model Backup Script - Upload trained model to Hugging Face Hub.

This script uploads your trained model to Hugging Face Hub for free cloud storage.
Your model will be private by default and can be easily restored later.

Usage:
    # First time setup (login to Hugging Face)
    python model_backup.py --login
    
    # Backup your model
    python model_backup.py --model outputs/run_name/final_model --repo your-username/model-name
    
    # Backup with custom message
    python model_backup.py --model outputs/run_name/final_model --repo your-username/model-name --message "Trained on batch 5"
"""

import argparse
import sys
from pathlib import Path


def login_to_hub():
    """Interactive login to Hugging Face Hub."""
    try:
        from huggingface_hub import login
        print("\n" + "="*60)
        print("🔐 Hugging Face Hub Login")
        print("="*60)
        print("\nTo get your access token:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'write' access")
        print("3. Copy and paste it below\n")
        login()
        print("\n✅ Login successful! You can now backup your models.")
    except ImportError:
        print("❌ Error: huggingface_hub not installed.")
        print("   Run: pip install huggingface_hub")
        sys.exit(1)


def backup_model(model_path: str, repo_id: str, commit_message: str = None, private: bool = True):
    """
    Upload model to Hugging Face Hub.
    
    Args:
        model_path: Path to the model directory (e.g., outputs/run_name/final_model)
        repo_id: Hugging Face repo ID (e.g., your-username/mental-health-detector)
        commit_message: Optional commit message
        private: Whether to make the repo private (default: True)
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ Error: huggingface_hub not installed.")
        print("   Run: pip install huggingface_hub")
        sys.exit(1)
    
    model_path = Path(model_path)
    
    # Validate model path
    if not model_path.exists():
        print(f"❌ Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    required_files = ["config.json", "tokenizer_config.json"]
    for f in required_files:
        if not (model_path / f).exists():
            print(f"❌ Error: Required file not found: {model_path / f}")
            print("   Make sure you're pointing to a valid model directory.")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("📤 Model Backup to Hugging Face Hub")
    print("="*60)
    print(f"\n📁 Model path: {model_path}")
    print(f"🏠 Repository: {repo_id}")
    print(f"🔒 Private: {private}")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        print(f"\n📦 Creating/checking repository...")
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"   ✅ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        sys.exit(1)
    
    # Upload the model
    try:
        print(f"\n⬆️  Uploading model files...")
        commit_message = commit_message or f"Model backup from {model_path.name}"
        
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        
        print(f"\n✅ Backup complete!")
        print(f"   🔗 View at: https://huggingface.co/{repo_id}")
        print(f"\n📥 To restore, run:")
        print(f"   python model_restore.py --repo {repo_id} --output outputs/restored_model")
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        sys.exit(1)


def list_models():
    """List available models in outputs directory."""
    outputs_path = Path("outputs")
    if not outputs_path.exists():
        print("❌ No outputs directory found.")
        return
    
    print("\n" + "="*60)
    print("📋 Available Models")
    print("="*60)
    
    found = False
    for run_dir in sorted(outputs_path.iterdir()):
        if run_dir.is_dir():
            final_model = run_dir / "final_model"
            if final_model.exists():
                # Get model size
                size = sum(f.stat().st_size for f in final_model.rglob("*") if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"\n  📦 {run_dir.name}")
                print(f"     Path: {final_model}")
                print(f"     Size: {size_mb:.1f} MB")
                found = True
    
    if not found:
        print("\n  No trained models found.")
        print("  Train a model first with: python train.py --config configs/incremental_batch.yaml --batch-number 1")


def main():
    parser = argparse.ArgumentParser(
        description="Backup trained model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Login to Hugging Face (first time only)
  python model_backup.py --login
  
  # List available models
  python model_backup.py --list
  
  # Backup a model
  python model_backup.py --model outputs/run_name/final_model --repo username/model-name
  
  # Backup with commit message
  python model_backup.py --model outputs/run_name/final_model --repo username/model-name --message "v1.0"
        """
    )
    
    parser.add_argument("--login", action="store_true", help="Login to Hugging Face Hub")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, help="Path to model directory")
    parser.add_argument("--repo", type=str, help="Hugging Face repo ID (e.g., username/model-name)")
    parser.add_argument("--message", type=str, help="Commit message for the backup")
    parser.add_argument("--public", action="store_true", help="Make the repository public (default: private)")
    
    args = parser.parse_args()
    
    if args.login:
        login_to_hub()
    elif args.list:
        list_models()
    elif args.model and args.repo:
        backup_model(args.model, args.repo, args.message, private=not args.public)
    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   1. python model_backup.py --login")
        print("   2. python model_backup.py --list")
        print("   3. python model_backup.py --model <path> --repo <username/name>")


if __name__ == "__main__":
    main()