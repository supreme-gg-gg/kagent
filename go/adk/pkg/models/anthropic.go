package models

import (
	"context"
	"fmt"
	"net/http"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/go-logr/logr"
	"golang.org/x/oauth2/google"
	googleoption "google.golang.org/api/option"
	googletransport "google.golang.org/api/transport/http"
)

// anthropicPassthroughOpts returns a per-request option that sets the Anthropic API key
// from the bearer token in ctx when APIKeyPassthrough is enabled. The Anthropic SDK sends
// this as the x-api-key header, which is the correct auth mechanism for Anthropic.
func anthropicPassthroughOpts(ctx context.Context, cfg *AnthropicConfig) []option.RequestOption {
	if !cfg.APIKeyPassthrough {
		return nil
	}
	if token, ok := ctx.Value(BearerTokenKey).(string); ok && token != "" {
		return []option.RequestOption{option.WithAPIKey(token)}
	}
	return nil
}

// AnthropicConfig holds Anthropic configuration
type AnthropicConfig struct {
	TransportConfig
	Model       string
	BaseUrl     string // Optional: override API base URL
	MaxTokens   *int
	Temperature *float64
	TopP        *float64
	TopK        *int
}

// AnthropicModel implements model.LLM for Anthropic Claude models.
type AnthropicModel struct {
	Config *AnthropicConfig
	Client anthropic.Client
	Logger logr.Logger
}

// NewAnthropicModelWithLogger creates a new Anthropic model instance with a logger
func NewAnthropicModelWithLogger(config *AnthropicConfig, logger logr.Logger) (*AnthropicModel, error) {
	apiKey := "passthrough" // placeholder; real auth set per-request by transport
	if !config.APIKeyPassthrough {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY environment variable is not set")
		}
	}
	return newAnthropicModelFromConfig(config, apiKey, logger)
}

func newAnthropicModelFromConfig(config *AnthropicConfig, apiKey string, logger logr.Logger) (*AnthropicModel, error) {
	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	// Set base URL if provided (useful for proxies or custom endpoints)
	if config.BaseUrl != "" {
		opts = append(opts, option.WithBaseURL(config.BaseUrl))
	}

	// Create HTTP client with TLS, custom headers, and timeout.
	httpClient, err := BuildHTTPClient(config.TransportConfig)
	if err != nil {
		return nil, err
	}
	if len(config.Headers) > 0 && logger.GetSink() != nil {
		logger.Info("Setting default headers for Anthropic client", "headersCount", len(config.Headers))
	}
	opts = append(opts, option.WithHTTPClient(httpClient))

	client := anthropic.NewClient(opts...)
	if logger.GetSink() != nil {
		logger.Info("Initialized Anthropic model", "model", config.Model, "baseUrl", config.BaseUrl)
	}

	return &AnthropicModel{
		Config: config,
		Client: client,
		Logger: logger,
	}, nil
}

// NewAnthropicVertexAIModelWithLogger creates an Anthropic model that authenticates
// via Google Cloud Vertex AI using Application Default Credentials (ADC).
// This is used for the GeminiAnthropic / AnthropicVertexAI provider type.
func NewAnthropicVertexAIModelWithLogger(ctx context.Context, config *AnthropicConfig, region, projectID string, logger logr.Logger) (*AnthropicModel, error) {
	creds, err := google.FindDefaultCredentials(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, fmt.Errorf("failed to find default credentials for Anthropic Vertex AI: %w", err)
	}

	baseClient, err := BuildHTTPClientWithoutHeaders(config.TransportConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to build HTTP client for Anthropic Vertex AI: %w", err)
	}
	authTransport, err := googletransport.NewTransport(
		ctx,
		baseClient.Transport,
		googleoption.WithTokenSource(creds.TokenSource),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to build authenticated transport for Anthropic Vertex AI: %w", err)
	}
	baseClient.Transport = WithHeaderTransport(authTransport, config.Headers)

	opts := []option.RequestOption{
		vertex.WithCredentials(ctx, region, projectID, creds),
		// vertex.WithCredentials configures the Vertex base URL and request
		// rewrite middleware. This client preserves that ADC auth behavior while
		// using kagent's TLS settings underneath it.
		option.WithHTTPClient(&http.Client{
			Timeout:   baseClient.Timeout,
			Transport: baseClient.Transport,
		}),
	}

	client := anthropic.NewClient(opts...)
	logger.Info("Initialized Anthropic Vertex AI model", "model", config.Model, "region", region, "project", projectID)

	return &AnthropicModel{
		Config: config,
		Client: client,
		Logger: logger,
	}, nil
}

// NewAnthropicBedrockModelWithLogger creates an Anthropic model that uses
// AWS Bedrock as the backend. Authentication is handled by the AWS SDK:
//   - If AWS_BEARER_TOKEN_BEDROCK is set, bearer token auth is used.
//   - Otherwise, standard AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
//     AWS_SESSION_TOKEN) or IAM roles are used via SigV4 signing.
//
// The region must be provided (e.g. "us-east-1") and determines the Bedrock endpoint.
func NewAnthropicBedrockModelWithLogger(ctx context.Context, config *AnthropicConfig, region string, logger logr.Logger) (*AnthropicModel, error) {
	opts := []option.RequestOption{
		bedrock.WithLoadDefaultConfig(ctx,
			awsconfig.WithRegion(region),
		),
	}

	// Create HTTP client with timeout, custom headers, TLS, and passthrough
	httpClient, err := BuildHTTPClient(config.TransportConfig)
	if err != nil {
		return nil, err
	}
	opts = append(opts, option.WithHTTPClient(httpClient))

	client := anthropic.NewClient(opts...)
	logger.Info("Initialized Anthropic Bedrock model", "model", config.Model, "region", region)

	return &AnthropicModel{
		Config: config,
		Client: client,
		Logger: logger,
	}, nil
}
