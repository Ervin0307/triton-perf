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
   input_file = "/root/kubernetes_files/triton-perf/output/2024-11-05T17:41:46Z.json"
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