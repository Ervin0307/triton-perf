package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/labstack/echo-contrib/echoprometheus"
	"github.com/spf13/viper"

	"triton-benchmark/handlers"
	"triton-benchmark/pkg"
)

func main() {
	logger := pkg.NewLogger()
	f, err := os.Stat("config.env")
	logger.Infof("%#v\n", f)
	logger.Info("Checking if .env file exists", err)
	// check if .env file exists
	if !os.IsNotExist(err) {
		logger.Debug("Loading .env file")
		viper.SetConfigFile("config.env")
		err := viper.ReadInConfig()
		if err != nil {
			logger.Error("Error loading .env file", err.Error())
			return
		}

	}

	h := handlers.NewAPIHandlerConfig(logger)

	h.EchoH.GET("/metrics", echoprometheus.NewHandler())
	h.EchoH.POST("/benchmark", h.BenchmarkHandler)
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		if err := h.EchoH.Start(":11000"); err != nil && err != http.ErrServerClosed {
			logger.Errorf("Error starting server: %v", err)
		}
	}()
	<-stop
	logger.Info("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := h.EchoH.Shutdown(ctx); err != nil {
		logger.Errorf("Error during server shutdown: %v", err)
		return
	}

	logger.Info("Server stopped")

}
