package triton

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
	"unsafe"
	"strconv"

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
	audioBuffer *audio.IntBuffer
}

var (
	networkTimeout = 600.0
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

	if tc.audioBuffer == nil {
		tc.audioBuffer = &audio.IntBuffer{
			Data:   make([]int, 8192),
			Format: format,
		}
	} else {
		tc.audioBuffer.Format = format
	}

	audioArr := make([]float32, 0, 1000000)

	start := time.Now()
	for {
		n, err := decoder.PCMBuffer(tc.audioBuffer)
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, 0, err
		}

		for i := 0; i < n; i++ {
			// Normalize audio data
			// diving by 32768.0 to normalize the audio data to [-1, 1]
			audioArr = append(audioArr, float32(tc.audioBuffer.Data[i])/32768.0)
		}

		if n < len(tc.audioBuffer.Data) {
			break
		}
	}
	tc.Log.Infof("Total Time taken for reading audio: %v secs\n", time.Since(start).Seconds())

	return audioArr, int(format.SampleRate), nil
}

func (tc *TritonClient) processAudio(audioFile string) error {
	audioArr, _, err := tc.readAudio(audioFile)
	if err != nil {
		tc.Log.Error("Error reading audio file: ", err)
		return err
	}

	samplingRateArr := []uint16{18000}

	request := &grpcClient.ModelInferRequest{
		ModelName: "whisper",
		Inputs: []*grpcClient.ModelInferRequest_InferInputTensor{
			{
				Name:     "INPUT0",
				Datatype: "FP32",
				Shape:    []int64{int64(len(audioArr))},
			},
			{
				Name:     "INPUT1",
				Datatype: "UINT16",
				Shape:    []int64{int64(len(samplingRateArr))},
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

	start := time.Now()
	response, err := tc.client.ModelInfer(ctx, request)
	if err != nil {
		tc.Log.Error("Error processing InferRequest: ", err)
		return err
	}
	tc.Log.Infof("Total Time taken for Inferece: %v secs\n", time.Since(start).Seconds())

	if len(response.RawOutputContents) > 0 {
		tc.Log.Infof("Output: %s\n", response.RawOutputContents[0])
	}

	return nil
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

	// var wg sync.WaitGroup
	// semaphore := make(chan struct{}, 5) // Limit concurrent goroutines, TODO: Make it configurable

	// for _, audioFile := range audioFiles {
	// 	wg.Add(1)
	// 	semaphore <- struct{}{}
	// 	go func(file string) {
	// 		defer wg.Done()
	// 		defer func() { <-semaphore }()

	// 		start := time.Now()
	// 		err := tc.processAudio(file)
	// 		if err != nil {
	// 			tc.Log.Errorf("Error processing %s: %v\n", file, err)
	// 		}
	// 		tc.Log.Infof("Total Time taken for %s: %v secs\n", file, time.Since(start).Seconds())
	// 	}(audioFile)
	// }

	// wg.Wait()
	// return nil

	// Get concurrency from environment variable, default to 5 if not set
	concurrency, err := strconv.Atoi(os.Getenv("TRITON_CONCURRENCY"))
	if err != nil || concurrency < 1 {
		concurrency = 5
	}

	tc.Log.Infof("Running benchmark with concurrency: %d\n", concurrency)

	var wg sync.WaitGroup
	wg.Add(concurrency)

	for i := 0; i < concurrency; i++ {
		go func() {
			defer wg.Done()

			// Select a random audio file
			file := audioFiles[i%len(audioFiles)]

			start := time.Now()
			err := tc.processAudio(file)
			if err != nil {
				tc.Log.Errorf("Error processing %s: %v\n", file, err)
			}
			tc.Log.Infof("Total Time taken for %s: %v secs\n", file, time.Since(start).Seconds())
		}()
	}

	wg.Wait()
	return nil
}
