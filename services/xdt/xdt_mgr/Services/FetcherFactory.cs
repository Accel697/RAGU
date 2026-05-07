using xdt_api.Interfaces;
using xdt_api.Sources.GitHub;

namespace xdt_mgr.Services;

public static class FetcherFactory
{
    public static ISourceFetcher Create(string sourceName, HttpClient http) => sourceName.ToLower() switch
    {
        "github" => new GitHubFetcher(http),
        _ => throw new NotSupportedException($"Fetcher for source '{sourceName}' is not supported")
    };
}