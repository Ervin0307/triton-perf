# Triton Perf

## Setup Instructions

1. Start Triton server
2. Update Triton GRPC host and port in config.env
3. Set the preferred concurrency number in config.env
4. Run `python preprocessing.py`
5. Run `make app`
6. To benchmark run `curl -v localhost:8000/benchmark -d '{}'`
7. Update the `input_file` variable (on line 87) in postprocessing.py to the output file path from step 4
8. Run `python postprocessing.py` to generate the `cleaned_output.csv` file that has accuracy metrics