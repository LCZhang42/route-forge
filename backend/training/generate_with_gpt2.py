"""
Generate climbing paths using fine-tuned GPT-2 model.
"""

import torch
import argparse
import sys
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(str(Path(__file__).parent.parent))
from models.tokenizer import ClimbPathTokenizer


def parse_generated_text(text: str):
    """
    Parse generated text back to grade and holds.
    
    Expected format: "GRADE: 7A | START: (0,4) | END: (8,17) | MID: (1,7) (3,11) (5,13)"
    
    Args:
        text: Generated text
        
    Returns:
        (grade, full_path) tuple or None if parsing fails
    """
    try:
        import re
        
        # Extract components
        grade_match = re.search(r'GRADE:\s*([^\|]+)', text)
        start_match = re.search(r'START:\s*\((\d+),(\d+)\)', text)
        end_match = re.search(r'END:\s*\((\d+),(\d+)\)', text)
        mid_match = re.search(r'MID:\s*([^\|]*?)(?:\||$)', text)
        
        if not grade_match or not start_match or not end_match:
            return None
        
        grade = grade_match.group(1).strip()
        start_hold = [int(start_match.group(1)), int(start_match.group(2))]
        end_hold = [int(end_match.group(1)), int(end_match.group(2))]
        
        # Parse intermediate holds
        mid_holds = []
        if mid_match:
            mid_str = mid_match.group(1).strip()
            if mid_str:
                matches = re.findall(r'\((\d+),(\d+)\)', mid_str)
                for match in matches:
                    x, y = int(match[0]), int(match[1])
                    mid_holds.append([x, y])
        
        # Construct full path: start + intermediate + end
        full_path = [start_hold] + mid_holds + [end_hold]
        
        return grade, full_path
    except Exception as e:
        print(f"Error parsing text: {e}")
        return None


def generate_paths(
    model_path: str,
    grade: str = None,
    start_hold: list = None,
    end_hold: list = None,
    num_samples: int = 5,
    max_length: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generate climbing paths using fine-tuned GPT-2 with endpoint conditioning.
    
    Args:
        model_path: Path to fine-tuned model
        grade: Optional grade to condition on (e.g., '7A')
        start_hold: Optional start hold [x, y] (e.g., [0, 4])
        end_hold: Optional end hold [x, y] (e.g., [8, 17])
        num_samples: Number of paths to generate
        max_length: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device to use
    """
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print()
    
    # Prepare prompt with endpoint conditioning
    if grade and start_hold and end_hold:
        prompt = f"GRADE: {grade} | START: ({start_hold[0]},{start_hold[1]}) | END: ({end_hold[0]},{end_hold[1]}) | MID:"
    elif grade:
        # Random endpoints for given grade
        import random
        start_hold = [random.randint(0, 10), random.randint(1, 6)]
        end_hold = [random.randint(0, 10), random.randint(12, 17)]
        prompt = f"GRADE: {grade} | START: ({start_hold[0]},{start_hold[1]}) | END: ({end_hold[0]},{end_hold[1]}) | MID:"
    else:
        # Fully random
        prompt = "GRADE:"
    
    print(f"Prompt: '{prompt}'")
    print(f"Generating {num_samples} paths...")
    print("="*60)
    
    valid_paths = []
    
    for i in range(num_samples):
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
            )
        
        # Decode
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        print(f"\nSample {i+1}:")
        print(f"  Generated: {generated_text}")
        
        # Parse
        parsed = parse_generated_text(generated_text)
        if parsed:
            gen_grade, full_path = parsed
            print(f"  Parsed Grade: {gen_grade}")
            print(f"  Start Hold: {full_path[0]}")
            print(f"  End Hold: {full_path[-1]}")
            print(f"  Intermediate Holds: {full_path[1:-1]}")
            print(f"  Total holds: {len(full_path)}")
            valid_paths.append((gen_grade, full_path))
        else:
            print("  Failed to parse")
    
    print("\n" + "="*60)
    print(f"Successfully generated {len(valid_paths)}/{num_samples} valid paths")
    
    return valid_paths


def main():
    parser = argparse.ArgumentParser(description='Generate climbing paths with GPT-2')
    
    parser.add_argument('--model_path', type=str, 
                        default='checkpoints/distilgpt2_climb/final',
                        help='Path to fine-tuned model')
    parser.add_argument('--grade', type=str, default=None,
                        help='Grade to condition on (e.g., 7A)')
    parser.add_argument('--start_hold', type=str, default=None,
                        help='Start hold as "x,y" (e.g., "0,4")')
    parser.add_argument('--end_hold', type=str, default=None,
                        help='End hold as "x,y" (e.g., "8,17")')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of paths to generate')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Nucleus sampling parameter')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Parse start/end holds if provided
    start_hold = None
    end_hold = None
    if args.start_hold:
        parts = args.start_hold.split(',')
        start_hold = [int(parts[0]), int(parts[1])]
    if args.end_hold:
        parts = args.end_hold.split(',')
        end_hold = [int(parts[0]), int(parts[1])]
    
    generate_paths(
        model_path=args.model_path,
        grade=args.grade,
        start_hold=start_hold,
        end_hold=end_hold,
        num_samples=args.num_samples,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )


if __name__ == '__main__':
    main()
