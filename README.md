# Triton Perf

## Setup Instructions

1. Update Triton GRPC host and port in config.env
2. Set the prferred concurrency number in config.env
3. Run `make app`
4. To benchmark run `curl -v localhost:8000/benchmark -d '{}'`