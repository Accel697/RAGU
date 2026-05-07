using System.Text.Json;
using System.Text.Json.Serialization;
using xdt_api.Models;

namespace xdt_mgr.Services;

public static class ConfigLoader
{
    private static readonly JsonSerializerOptions _options = new()
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public static CollectorConfig Load(string path)
    {
        var json = File.ReadAllText(path);
        Console.WriteLine($"[CONFIG] Loading: {path}");
        
        var config = JsonSerializer.Deserialize<CollectorConfig>(json, _options);
        
        if (config == null)
            throw new InvalidOperationException("Failed to parse config");

        foreach (var src in config.Sources)
        {
            Console.WriteLine($"[CONFIG] Source: '{src.Name}', URL: '{src.SourceUrl}', Method: '{src.FetchMethod}'");
            foreach (var prod in src.Products)
            {
                Console.WriteLine($"[CONFIG]   Product: '{prod.Name}', ReleasesUrl: '{prod.ReleasesUrl}'");
            }
        }

        return config;
    }
}