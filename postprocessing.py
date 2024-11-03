import json
import jiwer
from datasets import load_dataset

avg_acc = 0

def clean_json_outputs(input_file, output_file):
    """
    Read JSON file, clean the output fields by removing special characters,
    and save to a new JSON file.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to save cleaned JSON file
    """
    dataset_name = 'mozilla-foundation/common_voice_11_0'
    global avg_acc
    
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Clean each output field
    for key in data:
        # Get the output string
        output = data[key]['output']
        
        # Remove the special characters at the beginning
        # Find the first regular character (after the UTF-8 control characters)
        clean_start = 4  # Skip the first 4 characters (control sequence)
        cleaned_output = output[clean_start:]
        
        # Update the output field with cleaned text
        data[key]['output'] = cleaned_output.strip()

        acc = jiwer.wer(data[key]['actual_output'], data[key]['output'])
        avg_acc = avg_acc + acc
        data[key]['accuracy'] = acc

    
    # Save the cleaned data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, ensure_ascii=False)
    
    return data
  
def reverse_transform_filename_to_path(input_path):
    """
    Transform a filename back to its original path structure by:
    1. Removing the base directory (handlers/triton/audio/)
    2. Splitting the remaining path by '-' up to the last component
    3. Reconstructing the huggingface cache path
    4. Converting .wav to .mp3
    
    Args:
        input_path (str): The input path with transformed filename
        
    Returns:
        tuple: (directory_path, filename)
    """
    # Remove the base directory if it exists
    if input_path.startswith('handlers/triton/audio/'):
        filename = input_path.replace('handlers/triton/audio/', '')
    else:
        filename = input_path
    
    # Split the filename by '-'
    parts = filename.split('-')
    
    # The hash is the first part
    hash_value = parts[0]
    
    # The second to last part is the directory
    directory = parts[-2]
    
    # The last part is the filename (need to change extension)
    file_name = parts[-1].replace('.wav', '.mp3')
    
    # Construct the full directory path
    directory_path = f"/root/.cache/huggingface/datasets/downloads/extracted/{hash_value}/{directory}/{file_name}"
    
    return directory_path


# Example usage
if __name__ == "__main__":
    input_file = "/root/whisper_large_v2/output/2024-10-28T11:09:28Z.json"
    output_file = "cleaned_output.json"

    try:
        cleaned_data = clean_json_outputs(input_file, output_file)
        print(avg_acc / 1000)
        
            
        print(f"\nProcessing complete. Cleaned data saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in input file")
    except Exception as e:
        print(f"An error occurred: {str(e)}")