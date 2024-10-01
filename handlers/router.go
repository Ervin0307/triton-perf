package handlers

import (
	"triton-benchmark/handlers/triton"
	"triton-benchmark/pkg"

	"github.com/labstack/echo-contrib/echoprometheus"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

type APIHandlerConfig struct {
	log *pkg.Logger

	EchoH *echo.Echo

	TritonC *triton.TritonClient
}

func newEchoRouter() *echo.Echo {
	e := echo.New()
	e.Use(middleware.Logger())
	e.Use(middleware.Recover())
	e.Use(middleware.RequestID())
	e.Use(middleware.CORSWithConfig((middleware.CORSConfig{
		AllowOrigins: []string{"*"},
		AllowHeaders: []string{echo.HeaderOrigin, echo.HeaderContentType, echo.HeaderAccept, echo.HeaderAuthorization},
		AllowMethods: []string{echo.GET, echo.PUT, echo.PUT, echo.DELETE, echo.PATCH},
	})))
	e.Use(echoprometheus.NewMiddleware("triton_benchmark")) // adds middleware to gather metrics
	return e
}

func NewAPIHandlerConfig(loggerHandler *pkg.Logger) (*APIHandlerConfig) {
	return &APIHandlerConfig{
		log:   loggerHandler,
		EchoH: newEchoRouter(),
		TritonC: &triton.TritonClient{
			Log: loggerHandler,
		},
	}
}
