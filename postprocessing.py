# # import json
# # import jiwer
# # from datasets import load_dataset

# # avg_acc = 0

# # # def find_audio_by_path(dataset_name, target_path):
# # #     """
# # #     Find a specific audio file in the dataset based on its path.
    
# # #     Args:
# # #         dataset_name (str): Name of the dataset
# # #         split (str): Dataset split (e.g., 'test', 'train')
# # #         target_path (str): The path of the audio file to find
        
# # #     Returns:
# # #         dict: The matching dataset entry or None if not found
# # #     """
# # #     # Load the dataset
# # #     dataset = load_dataset(dataset_name, "en", split="test", trust_remote_code=True)
    
# # #     # Filter the dataset to find the matching entry
# # #     filtered_dataset = dataset.filter(lambda x: x['path'] == target_path)
    
# # #     # Return the first (and should be only) matching entry if found
# # #     if len(filtered_dataset) > 0:
# # #         return filtered_dataset[0]
# # #     return None

# # def clean_json_outputs(input_file, output_file):
# #     """
# #     Read JSON file, clean the output fields by removing special characters,
# #     and save to a new JSON file.
    
# #     Args:
# #         input_file (str): Path to input JSON file
# #         output_file (str): Path to save cleaned JSON file
# #     """
# #     dataset_name = 'mozilla-foundation/common_voice_11_0'
# #     global avg_acc
    
# #     # Read the JSON file
# #     with open(input_file, 'r') as f:
# #         data = json.load(f)
    
# #     # Clean each output field
# #     for key in data:
# #         # Get the output string
# #         output = data[key]['output']
        
# #         # Remove the special characters at the beginning
# #         # Find the first regular character (after the UTF-8 control characters)
# #         clean_start = 4  # Skip the first 4 characters (control sequence)
# #         cleaned_output = output[clean_start:]
        
# #         # Update the output field with cleaned text
# #         data[key]['output'] = cleaned_output.strip()
# #         # data[key]['file_name'] = reverse_transform_filename_to_path(data[key]['file_name'])
# #         # dataset = find_audio_by_path(dataset_name=dataset_name, target_path=data[key]['file_name'])
# #         # if dataset is None:
# #         #     continue        

# #         acc = jiwer.wer(data[key]['actual_output'], data[key]['output'])
# #         avg_acc = avg_acc + acc
# #         data[key]['accuracy'] = acc

    
# #     # Save the cleaned data to a new JSON file
# #     with open(output_file, 'w') as f:
# #         json.dump(data, f, ensure_ascii=False)
    
# #     return data
  
# # def reverse_transform_filename_to_path(input_path):
# #     """
# #     Transform a filename back to its original path structure by:
# #     1. Removing the base directory (handlers/triton/audio/)
# #     2. Splitting the remaining path by '-' up to the last component
# #     3. Reconstructing the huggingface cache path
# #     4. Converting .wav to .mp3
    
# #     Args:
# #         input_path (str): The input path with transformed filename
        
# #     Returns:
# #         tuple: (directory_path, filename)
# #     """
# #     # Remove the base directory if it exists
# #     if input_path.startswith('handlers/triton/audio/'):
# #         filename = input_path.replace('handlers/triton/audio/', '')
# #     else:
# #         filename = input_path
    
# #     # Split the filename by '-'
# #     parts = filename.split('-')
    
# #     # The hash is the first part
# #     hash_value = parts[0]
    
# #     # The second to last part is the directory
# #     directory = parts[-2]
    
# #     # The last part is the filename (need to change extension)
# #     file_name = parts[-1].replace('.wav', '.mp3')
    
# #     # Construct the full directory path
# #     directory_path = f"/root/.cache/huggingface/datasets/downloads/extracted/{hash_value}/{directory}/{file_name}"
    
# #     return directory_path

# # # if __name__ == "__main__":
# # #     # Test path
# # #     test_path = "handlers/triton/audio/108568e2b1202910baba7c2a087a0e223771ccfa6cbc287844372f3d78bdb6f2-en_test_0-common_voice_en_27710027.wav"
    
# # #     # Transform the path
# # #     dir_path, filename = reverse_transform_filename_to_path(test_path)
    
# # #     print("Original transformed path:", test_path)
# # #     print("\nReversed transformation:")
# # #     print("Directory path:", dir_path)
# # #     print("Filename:", filename)
# # #     print("\nFull path:", f"{dir_path}/{filename}")

# # # # Example output:
# # # # Directory path: /root/.cache/huggingface/datasets/downloads/extracted/108568e2b1202910baba7c2a087a0e223771ccfa6cbc287844372f3d78bdb6f2/en_test_0
# # # # Filename: common_voice_en_27710027.mp3

# # # Example usage
# # if __name__ == "__main__":
# #     input_file = "/root/kubernetes_files/whispher_large_v2/output/2024-11-05T11:08:28Z.json"
# #     output_file = "cleaned_output.json"

# #     try:
# #         cleaned_data = clean_json_outputs(input_file, output_file)
# #         print(avg_acc / 1000)
        
# #         # Print example of cleaned outputs
# #         # print("Sample of cleaned outputs:")
# #         # for i in range(min(3, len(cleaned_data))):
# #         #   print(f"{i}: {cleaned_data[str(i)]['output']}")
            
# #         print(f"\nProcessing complete. Cleaned data saved to {output_file}")
        
# #     except FileNotFoundError:
# #         print(f"Error: Could not find input file {input_file}")
# #     except json.JSONDecodeError:
# #         print("Error: Invalid JSON format in input file")
# #     except Exception as e:
# #         print(f"An error occurred: {str(e)}")


# import json
# import jiwer
# from datasets import load_dataset
# from datetime import datetime
# import dateutil.parser
# import numpy as np  # For statistical calculations

# avg_acc = 0

# def calculate_metrics(data):
#     """
#     Calculate throughput and latency metrics.
    
#     Args:
#         data (dict): JSON data containing request timing information
    
#     Returns:
#         tuple: (throughput, total_time, latency_metrics)
#     """
#     # Calculate timestamps for throughput
#     start_times = [dateutil.parser.parse(data[str(i)]['start_time']) 
#                   for i in range(len(data))]
#     end_times = [dateutil.parser.parse(data[str(i)]['end_time']) 
#                 for i in range(len(data))]
    
#     # Calculate overall throughput metrics
#     overall_start = min(start_times)
#     overall_end = max(end_times)
#     total_time = (overall_end - overall_start).total_seconds()
#     throughput = len(data) / total_time if total_time > 0 else 0
    
#     # Calculate latency for each request
#     latencies = []
#     for i in range(len(data)):
#         start = dateutil.parser.parse(data[str(i)]['start_time'])
#         end = dateutil.parser.parse(data[str(i)]['end_time'])
#         latency = (end - start).total_seconds() * 1000  # Convert to milliseconds
#         latencies.append(latency)
#         # Add latency to the data dictionary
#         data[str(i)]['latency_ms'] = latency
    
#     # Calculate latency statistics
#     latency_metrics = {
#         'mean_latency_ms': np.mean(latencies),
#         'median_latency_ms': np.median(latencies),
#         'p95_latency_ms': np.percentile(latencies, 95),
#         'p99_latency_ms': np.percentile(latencies, 99),
#         'min_latency_ms': np.min(latencies),
#         'max_latency_ms': np.max(latencies),
#         'std_latency_ms': np.std(latencies)
#     }
    
#     return throughput, total_time, latency_metrics

# def clean_json_outputs(input_file, output_file):
#     """
#     Read JSON file, clean the output fields, calculate metrics,
#     and save to a new JSON file.
#     """
#     dataset_name = 'mozilla-foundation/common_voice_11_0'
#     global avg_acc
    
#     # Read the JSON file
#     with open(input_file, 'r') as f:
#         data = json.load(f)
    
#     # Calculate performance metrics
#     throughput, total_time, latency_metrics = calculate_metrics(data)
    
#     # Clean each output field
#     for key in data:
#         if key.isdigit():  # Only process numbered entries
#             output = data[key]['output']
#             clean_start = 4
#             cleaned_output = output[clean_start:]
#             data[key]['output'] = cleaned_output.strip()
            
#             acc = jiwer.wer(data[key]['actual_output'], data[key]['output'])
#             avg_acc = avg_acc + acc
#             data[key]['accuracy'] = acc

#     # Add performance metrics to the output
#     data['metrics'] = {
#         'throughput_qps': throughput,
#         'total_time_seconds': total_time,
#         'total_requests': len(data), 
#         'average_accuracy': avg_acc / (len(data) - 1),
#         **latency_metrics  # Include all latency metrics
#     }
    
#     # Save the cleaned data to a new JSON file
#     with open(output_file, 'w') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)
    
#     return data, throughput, total_time, latency_metrics

# if __name__ == "__main__":
#     input_file = "/root/kubernetes_files/whispher_large_v2/output/2024-11-05T11:08:28Z.json"
#     output_file = "cleaned_output.json"

#     try:
#         cleaned_data, throughput, total_time, latency_metrics = clean_json_outputs(input_file, output_file)
        
#         print("\nPerformance Metrics:")
#         print(f"Throughput: {throughput:.2f} requests/second")
#         print(f"Total Time: {total_time:.2f} seconds")
#         print(f"Total Requests: {len(cleaned_data) - 1}")  # Subtract 1 for metrics key
#         print(f"Average Accuracy: {avg_acc / (len(cleaned_data) - 1):.4f}")
        
#         print("\nLatency Metrics:")
#         print(f"Mean Latency: {latency_metrics['mean_latency_ms']:.2f} ms")
#         print(f"Median Latency: {latency_metrics['median_latency_ms']:.2f} ms")
#         print(f"P95 Latency: {latency_metrics['p95_latency_ms']:.2f} ms")
#         print(f"P99 Latency: {latency_metrics['p99_latency_ms']:.2f} ms")
#         print(f"Min Latency: {latency_metrics['min_latency_ms']:.2f} ms")
#         print(f"Max Latency: {latency_metrics['max_latency_ms']:.2f} ms")
#         print(f"Std Dev Latency: {latency_metrics['std_latency_ms']:.2f} ms")
        
#         print(f"\nProcessing complete. Cleaned data saved to {output_file}")
        
#     except FileNotFoundError:
#         print(f"Error: Could not find input file {input_file}")
#     except json.JSONDecodeError:
#         print("Error: Invalid JSON format in input file")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

import json
import jiwer
from datasets import load_dataset
from datetime import datetime
import dateutil.parser
import numpy as np

avg_wer = 0

def calculate_metrics(data):
   """
   Calculate throughput and latency metrics.
   """
   start_times = [dateutil.parser.parse(data[str(i)]['start_time']) 
                 for i in range(len(data))]
   end_times = [dateutil.parser.parse(data[str(i)]['end_time']) 
               for i in range(len(data))]
   
   overall_start = min(start_times)
   overall_end = max(end_times)
   total_time = (overall_end - overall_start).total_seconds()
   throughput = len(data) / total_time if total_time > 0 else 0
   
   latencies = []
   wer_scores = []
   for i in range(len(data)):
       start = dateutil.parser.parse(data[str(i)]['start_time'])
       end = dateutil.parser.parse(data[str(i)]['end_time'])
       latency = (end - start).total_seconds() * 1000  # Convert to milliseconds
       latencies.append(latency)
       wer_scores.append(data[str(i)].get('wer', 0))
       data[str(i)]['latency_ms'] = latency
   
   # Calculate average WER and WAR
   avg_wer = np.mean(wer_scores)
   avg_war = 1 - avg_wer

   latency_metrics = {
       'mean_latency_ms': np.mean(latencies),
       'median_latency_ms': np.median(latencies),
       'p95_latency_ms': np.percentile(latencies, 95),
       'p99_latency_ms': np.percentile(latencies, 99),
       'min_latency_ms': np.min(latencies),
       'max_latency_ms': np.max(latencies),
       'std_latency_ms': np.std(latencies)
   }
   
   return throughput, total_time, latency_metrics, avg_wer, avg_war

def clean_json_outputs(input_file, output_file):
   """
   Read JSON file, clean the output fields, calculate metrics,
   and save to a new JSON file.
   """
   dataset_name = 'mozilla-foundation/common_voice_11_0'
   global avg_wer
   
   with open(input_file, 'r') as f:
       data = json.load(f)

   wer_scores = []
   for key in data:
       if key.isdigit():
           output = data[key]['output']
           clean_start = 4
           cleaned_output = output[clean_start:]
           data[key]['output'] = cleaned_output.strip()
           
           wer = jiwer.wer(data[key]['actual_output'], data[key]['output'])
           war = 1 - wer
           data[key]['wer'] = wer
           data[key]['war'] = war
           wer_scores.append(wer)
   
   throughput, total_time, latency_metrics, avg_wer, avg_war = calculate_metrics(data)

   data['metrics'] = {
       'throughput_qps': throughput,
       'total_time_seconds': total_time,
       'total_requests': len(data) - 1,
       'average_wer': avg_wer,
       'average_war': avg_war,
       **latency_metrics
   }
   
   with open(output_file, 'w') as f:
       json.dump(data, f, ensure_ascii=False, indent=4)
   
   return data, throughput, total_time, latency_metrics, avg_wer, avg_war

if __name__ == "__main__":
   input_file = "/root/kubernetes_files/whispher_large_v2/output/2024-11-05T11:08:28Z.json"
   output_file = "cleaned_output.json"

   try:
       cleaned_data, throughput, total_time, latency_metrics, avg_wer, avg_war = clean_json_outputs(input_file, output_file)
       
       print("\nPerformance Metrics:")
       print(f"Throughput: {throughput:.2f} requests/second")
       print(f"Total Time: {total_time:.2f} seconds")
       print(f"Total Requests: {len(cleaned_data) - 1}")
       print(f"Average WER: {avg_wer:.4f} (lower is better)")
       print(f"Average WAR: {avg_war:.4f} (higher is better)")
       
       print("\nLatency Metrics:")
       print(f"Mean Latency: {latency_metrics['mean_latency_ms']:.2f} ms")
       print(f"Median Latency: {latency_metrics['median_latency_ms']:.2f} ms")
       print(f"P95 Latency: {latency_metrics['p95_latency_ms']:.2f} ms")
       print(f"P99 Latency: {latency_metrics['p99_latency_ms']:.2f} ms")
       print(f"Min Latency: {latency_metrics['min_latency_ms']:.2f} ms")
       print(f"Max Latency: {latency_metrics['max_latency_ms']:.2f} ms")
       print(f"Std Dev Latency: {latency_metrics['std_latency_ms']:.2f} ms")
       
       print(f"\nProcessing complete. Cleaned data saved to {output_file}")
       
   except FileNotFoundError:
       print(f"Error: Could not find input file {input_file}")
   except json.JSONDecodeError:
       print("Error: Invalid JSON format in input file")
   except Exception as e:
       print(f"An error occurred: {str(e)}")