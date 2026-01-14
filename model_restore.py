#!/usr/bin/env python3
"""
Model Restore Script - Download trained model from Hugging Face Hub.

This script downloads your model from Hugging Face Hub to use locally.

Usage:
    # Restore a model
    python model_restore.py --repo your-username/model-name
    
    # Restore to specific directory
    python model_restore.py --repo your-username/model-name --output outputs/my_model/final_model
    
    # List your models on Hugging Face
    python model_restore.py --list-remote
"""

import argparse
import sys
from pathlib import Path


def list_remote_models(username: str = None):
    """List models available on Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi, list_models
    except ImportError:
        print("❌ Error: huggingface_hub not installed.")
        print("   Run: pip install huggingface_hub")
        sys.exit(1)
    
    api = HfApi()
    
    print("\n" + "="*60)
    print("🌐 Your Models on Hugging Face Hub")
    print("="*60)
    
    try:
        # Get current user if not specified
        if not username:
            user_info = api.whoami()
            username = user_info.get("name")
            if not username:
                print("\n⚠️  Not logged in. Run: python model_backup.py --login")
                print("   Or specify username: python model_restore.py --list-remote --user USERNAME")
                return
        
        print(f"\n👤 User: {username}\n")
        
        # List user's models
        models = list(api.list_models(author=username))
        
        if not models:
            print("   No models found.")
            print("   Backup a model first with: python model_backup.py --model <path> --repo <name>")
            return
        
        for model in models:
            print(f"  📦 {model.id}")
            if model.private:
                print(f"     🔒 Private")
            else:
                print(f"     🌍 Public")
            print(f"     🔗 https://huggingface.co/{model.id}")
            print()
            
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        print("   Make sure you're logged in: python model_backup.py --login")


def restore_model(repo_id: str, output_path: str = None, revision: str = None):
    """
    Download model from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face repo ID (e.g., your-username/mental-health-detector)
        output_path: Where to save the model (default: outputs/<repo_name>/final_model)
        revision: Specific revision/commit to download (default: latest)
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ Error: huggingface_hub not installed.")
        print("   Run: pip install huggingface_hub")
        sys.exit(1)
    
    # Default output path
    if not output_path:
        repo_name = repo_id.split("/")[-1]
        output_path = f"outputs/{repo_name}/final_model"
    
    output_path = Path(output_path)
    
    print("\n" + "="*60)
    print("📥 Model Restore from Hugging Face Hub")
    print("="*60)
    print(f"\n🏠 Repository: {repo_id}")
    print(f"📁 Output path: {output_path}")
    if revision:
        print(f"📌 Revision: {revision}")
    
    # Check if output already exists
    if output_path.exists():
        print(f"\n⚠️  Output path already exists: {output_path}")
        response = input("   Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("   Cancelled.")
            return
    
    # Create parent directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"\n⬇️  Downloading model...")
        
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_path),
            revision=revision,
        )
        
        print(f"\n✅ Restore complete!")
        print(f"   📁 Model saved to: {output_path}")
        
        # Verify the download
        required_files = ["config.json", "tokenizer_config.json"]
        missing = [f for f in required_files if not (output_path / f).exists()]
        
        if missing:
            print(f"\n⚠️  Warning: Some expected files are missing: {missing}")
        else:
            print(f"\n🎉 Model is ready to use!")
            print(f"\n   To use in the app:")
            print(f"   1. Run: python app.py")
            print(f"   2. Select the model from the dropdown")
            print(f"\n   Or use directly:")
            print(f"   from predict import Predictor")
            print(f"   predictor = Predictor('{output_path}')")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("\n   Possible issues:")
        print("   - Repository doesn't exist")
        print("   - Repository is private and you're not logged in")
        print("   - Network connection issue")
        print("\n   Try: python model_backup.py --login")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Restore trained model from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List your models on Hugging Face
  python model_restore.py --list-remote
  
  # Restore a model (auto-creates outputs/<name>/final_model)
  python model_restore.py --repo username/model-name
  
  # Restore to specific directory
  python model_restore.py --repo username/model-name --output outputs/my_model/final_model
  
  # Restore specific version
  python model_restore.py --repo username/model-name --revision v1.0
        """
    )
    
    parser.add_argument("--list-remote", action="store_true", help="List your models on Hugging Face Hub")
    parser.add_argument("--user", type=str, help="Hugging Face username (for --list-remote)")
    parser.add_argument("--repo", type=str, help="Hugging Face repo ID (e.g., username/model-name)")
    parser.add_argument("--output", type=str, help="Output directory for the model")
    parser.add_argument("--revision", type=str, help="Specific revision/commit to download")
    
    args = parser.parse_args()
    
    if args.list_remote:
        list_remote_models(args.user)
    elif args.repo:
        restore_model(args.repo, args.output, args.revision)
    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   1. python model_restore.py --list-remote")
        print("   2. python model_restore.py --repo <username/model-name>")


if __name__ == "__main__":
    main()