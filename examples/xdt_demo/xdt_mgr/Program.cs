using System.Text.Json;
using xdt_api.Interfaces;
using xdt_api.Models;
using xdt_mgr.Services;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddHttpClient();

var app = builder.Build();

app.MapPost("/collect", async (IHttpClientFactory httpFactory) =>
{
    var configPath = Path.Combine(AppContext.BaseDirectory, "xdt_rpo.json");
    var config = ConfigLoader.Load(configPath);
    
    var results = new List<CollectionResult>();
    
    using var httpClient = httpFactory.CreateClient();
    httpClient.Timeout = TimeSpan.FromMinutes(2);

    foreach (var source in config.Sources)
    {
        ISourceFetcher fetcher;
        try
        {
            fetcher = FetcherFactory.Create(source.Name, httpClient);
        }
        catch (NotSupportedException ex)
        {
            results.Add(new CollectionResult
            {
                Source = source.Name,
                Product = "N/A",
                RawData = ex.Message,
                CollectedAt = DateTime.UtcNow
            });
            continue;
        }
        
        foreach (var product in source.Products)
        {
            try
            {
                var rawData = await fetcher.FetchAsync(source.SourceUrl, product.ReleasesUrl);
                
                results.Add(new CollectionResult
                {
                    Source = source.Name,
                    Product = product.Name,
                    RawData = rawData,
                    CollectedAt = DateTime.UtcNow
                });
            }
            catch (Exception ex)
            {
                results.Add(new CollectionResult
                {
                    Source = source.Name,
                    Product = product.Name,
                    RawData = $"Error: {ex.Message}",
                    CollectedAt = DateTime.UtcNow
                });
            }
        }
    }
    
    return Results.Json(results, new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower });
});

app.MapGet("/", () => "xdt_mgr is running. POST /collect to start collection.");

app.Run("http://0.0.0.0:8081");