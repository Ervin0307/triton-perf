package triton

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"time"
	"unsafe"

	grpcClient "triton-benchmark/handlers/triton/grpc-client"
	"triton-benchmark/pkg"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/spf13/viper"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
)

type TritonClient struct {
	Log *pkg.Logger

	conn        *grpc.ClientConn
	client      grpcClient.GRPCInferenceServiceClient
}

type Output struct {
	StartTime time.Time `json:"start_time"`
	EndTime time.Time `json:"end_time"`
	InferenceOutput string `json:"output"`
	Duration time.Duration `json:"duration"`
	FileName string `json:"file_name"`
}

var (
	networkTimeout = 500.0
)

func serverLiveRequest(client grpcClient.GRPCInferenceServiceClient) (*grpcClient.ServerLiveResponse, error) {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverLiveRequest := grpcClient.ServerLiveRequest{}
	// Submit ServerLive request to server
	serverLiveResponse, err := client.ServerLive(ctx, &serverLiveRequest)
	if err != nil {
		return nil, err
	}
	return serverLiveResponse, nil
}

func serverReadyRequest(client grpcClient.GRPCInferenceServiceClient) (*grpcClient.ServerReadyResponse, error) {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverReadyRequest := grpcClient.ServerReadyRequest{}
	// Submit ServerReady request to server
	serverReadyResponse, err := client.ServerReady(ctx, &serverReadyRequest)
	if err != nil {
		return nil, err
	}
	return serverReadyResponse, nil
}

func (tc *TritonClient) readAudio(audioFilePath string) ([]float32, int, error) {
	f, err := os.Open(audioFilePath)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	decoder := wav.NewDecoder(f)
	format := decoder.Format()

	// Create a new buffer for each call instead of sharing one
	audioBuffer := &audio.IntBuffer{
		Data:   make([]int, 8192),
		Format: format,
	}

	audioArr := make([]float32, 0, 1000000)

	// start := time.Now()
	for {
		n, err := decoder.PCMBuffer(audioBuffer)
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, 0, err
		}

		for i := 0; i < n; i++ {
			// Normalize audio data
			// diving by 32768.0 to normalize the audio data to [-1, 1]
			audioArr = append(audioArr, float32(audioBuffer.Data[i])/32768.0)
		}

		if n < len(audioBuffer.Data) {
			break
		}
	}
	// tc.Log.Infof("Total Time taken for reading audio: %v secs\n", time.Since(start).Seconds())

	return audioArr, int(format.SampleRate), nil
}

	type metadata struct {
		start time.Time
		duration time.Duration
		end time.Time
		output []byte
	}

func (tc *TritonClient) processAudio(audioFile string) (*metadata, error) {
	audioArr, _, err := tc.readAudio(audioFile)
	if err != nil {
		tc.Log.Error("Error reading audio file: ", err)
		return nil, err
	}

	// tc.Log.Infof("Processing audio file: %d\n", len(audioArr))

	samplingRateArr := []uint16{18000}

	request := &grpcClient.ModelInferRequest{
		ModelName: "whisper_batched",
		Inputs: []*grpcClient.ModelInferRequest_InferInputTensor{
			{
				Name:     "INPUT0",
				Datatype: "FP32",
				Shape:    []int64{1, int64(len(audioArr))},
			},
		},
		Outputs: []*grpcClient.ModelInferRequest_InferRequestedOutputTensor{
			{Name: "OUTPUT0"},
		},
		RawInputContents: [][]byte{
			float32SliceToBytes(audioArr),
			uint16SliceToBytes(samplingRateArr),
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(networkTimeout)*time.Second)
	defer cancel()

	metadata := metadata{}
	
	metadata.start = time.Now()
	response, err := tc.client.ModelInfer(ctx, request)
	if err != nil {
		tc.Log.Error("Error processing InferRequest: ", err)
		return nil, err
	}
	metadata.duration = time.Since(metadata.start)
	metadata.end = time.Now()
	// tc.Log.Infof("Total Time taken for Inferece: %v secs\n", time.Since(start).Seconds())

	if len(response.RawOutputContents) > 0 {
		metadata.output = response.RawOutputContents[0]
		tc.Log.Debugf("Output: %s\n", string(response.RawOutputContents[0]))
	}

	return &metadata, nil
}

func float32SliceToBytes(slice []float32) []byte {
	// Allocate a byte slice with the exact size needed
	bytes := make([]byte, len(slice)*4)

	// Use unsafe.Pointer to treat the float32 slice as a byte slice
	// This avoids the need for explicit conversion of each element and memory allocation
	// allocates an aribitary ptr to byte slice of size 1<<30 (1GB)
	byteSlice := (*[1 << 30]byte)(unsafe.Pointer(&slice[0]))[: len(slice)*4 : len(slice)*4]

	// Copy the bytes directly
	copy(bytes, byteSlice)

	return bytes
}

func uint16SliceToBytes(slice []uint16) []byte {
	// Allocate a byte slice with the exact size needed
	bytes := make([]byte, len(slice)*2)

	// Use unsafe.Pointer to treat the uint16 slice as a byte slice
	// This avoids the need for explicit conversion of each element and memory allocation
	// allocates an aribitary ptr to byte slice of size 1<<30 (1GB)
	byteSlice := (*[1 << 30]byte)(unsafe.Pointer(&slice[0]))[: len(slice)*2 : len(slice)*2]

	// Copy the bytes directly
	copy(bytes, byteSlice)

	return bytes
}

func (tc *TritonClient) Connect() error {
	// Connect to gRPC server
	URL := viper.GetString("TRITON_GRPC_URL")
	tc.Log.Infof("Connecting to Triton server at %s...", URL)
	conn, err := grpc.Dial(URL, grpc.WithInsecure())
	if err != nil {
		tc.Log.Errorf("Couldn't connect to endpoint %s: %v\n", URL, err)
		return err
	}
	tc.conn = conn

	// Create client from gRPC server connection
	tc.client = grpcClient.NewGRPCInferenceServiceClient(conn)

	return nil
}

func (tc *TritonClient) Close() {
	tc.conn.Close()
}

func (tc *TritonClient) GetConnectionState() connectivity.State {
	return tc.conn.GetState()
}


func (tc *TritonClient) Serve() error {
    // Server health checks...

    audioPath := viper.GetString("AUDIO_PATH")
    audioFiles, err := filepath.Glob(audioPath + "*.*")
    if err != nil {
        tc.Log.Errorf("Error getting audio files: %v\n", err)
        return err
    }

    concurrency, err := strconv.Atoi(viper.GetString("TRITON_CONCURRENCY"))
    if err != nil || concurrency < 1 {
        concurrency = 5
    }

    tc.Log.Infof("Running benchmark with concurrency: %d\n", concurrency)

    // Create a channel to collect timing results
    type processingResult struct {
        workerID  int
        file      string
        duration  time.Duration
        startTime time.Time
        endTime   time.Time
        err       error
				output  []byte
    }
    results := make(chan processingResult, concurrency)

    // Start the workers
    var wg sync.WaitGroup
    wg.Add(concurrency)

    for i := 0; i < concurrency; i++ {
        go func(workerID int) {
            defer wg.Done()

            file := audioFiles[workerID%len(audioFiles)]

            metadata, err := tc.processAudio(file)
            
            results <- processingResult{
                workerID:  workerID,
                file:      file,
                duration:  metadata.duration,
                startTime: metadata.start,
                endTime:   metadata.end,
                err:      err,
								output: metadata.output,
            }
        }(i)
    }

    // Start a goroutine to close results channel after all workers finish
    go func() {
        wg.Wait()
        close(results)
        tc.Log.Debug("All workers finished")
    }()

    // Collect and process results
    var totalDuration time.Duration
    var processedFiles int
    var durations []time.Duration // Slice to store all durations for percentile calculation

		outputs := make(map[int]Output, 0)

    for result := range results {
        processedFiles++
        totalDuration += result.duration

        durations = append(durations, result.duration) // Store duration for percentile calculation

				outputs[result.workerID] = Output{
					StartTime: result.startTime,
					EndTime: result.endTime,
					InferenceOutput: string(result.output),
					Duration: result.duration,
					FileName: result.file,
				}

        if result.err != nil {
            tc.Log.Errorf("Worker %d - Error processing %s: %v\n", 
                result.workerID, result.file, result.err)
        } else {
            tc.Log.Infof("Worker %d - Processed %s in %v (Started: %s, Finished: %s)\n",
                result.workerID, 
                result.file, 
                result.duration,
                result.startTime.Format("15:04:05.000"),
                result.endTime.Format("15:04:05.000"))
        }
    }

    // Calculate 99th percentile
    p99 := calculatePercentile(durations, 99)

    // Log summary statistics
    tc.Log.Infof("Summary Statistics:")
    tc.Log.Infof("Total files processed: %d", processedFiles)
    tc.Log.Infof("Average processing time: %v", totalDuration/time.Duration(processedFiles))
    tc.Log.Infof("99th percentile processing time: %v", p99)
    
    // Log individual worker statistics
    for workerID := 0; workerID < concurrency; workerID++ {
			if op, ok := outputs[workerID]; ok {
				tc.Log.Infof("Worker %d - Total running time: %v",  workerID, op.EndTime.Sub(op.StartTime))
			}
    }

		fmt.Println("length", len(outputs))

		err = saveResult(outputs)
		if err != nil {
			tc.Log.Errorf("Error saving results: %v\n", err)
		}

    return nil
}

func saveResult(outputs map[int]Output) error {
	
	if err := os.MkdirAll(viper.GetString("OUTPUT_PATH"), os.ModePerm); err != nil {
		return err
	}

	f, err := os.Create(filepath.Join(viper.GetString("OUTPUT_PATH"), fmt.Sprintf("%s.json", time.Now().Format(time.RFC3339))))
	if err != nil {
		return err
	}
	defer f.Close()

	as_json, _ := json.MarshalIndent(outputs, "", "\t")
	_, err = f.Write(as_json)
	if err != nil {
		return err
	}

	return nil
}

// calculatePercentile calculates the nth percentile from a slice of durations
func calculatePercentile(durations []time.Duration, n float64) time.Duration {
    if len(durations) == 0 {
        return 0
    }
    
    // Sort durations
    sort.Slice(durations, func(i, j int) bool {
        return durations[i] < durations[j]
    })
    
    // Calculate the index for the percentile
    index := int(math.Ceil((n/100) * float64(len(durations)))) - 1
    
    // Ensure index is within bounds
    if index < 0 {
        index = 0
    }
    if index >= len(durations) {
        index = len(durations) - 1
    }
    
    return durations[index]
}
