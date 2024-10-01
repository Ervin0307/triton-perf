package handlers

import (
	"github.com/labstack/echo/v4"
)

func (h *APIHandlerConfig) BenchmarkHandler(c echo.Context) error {
	h.log.Info("Running Triton Benchmark")

	err := h.TritonC.Connect()
	if err != nil {
		h.log.Errorf("Error connecting to Triton server: %v", err)
		return c.JSON(500, "Error connecting to Triton server")
	}
	defer h.TritonC.Close()

	err = h.TritonC.Serve()
	if err != nil {
		h.log.Errorf("Error serving Triton server: %v", err)
		return c.JSON(500, "Error serving Triton server")
	}

	return nil
}
