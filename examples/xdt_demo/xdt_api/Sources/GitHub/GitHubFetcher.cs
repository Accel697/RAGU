using System.Net.Http.Headers;
using xdt_api.Interfaces;
using xdt_api.Sources.GitHub;

namespace xdt_api.Sources.GitHub;

public class GitHubFetcher : ISourceFetcher
{
    private readonly HttpClient _http;

    public GitHubFetcher(HttpClient http)
    {
        _http = http;
        
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("xdt_demo/1.0");
        _http.DefaultRequestHeaders.Accept.ParseAdd("application/vnd.github+json");
        
        var token = Environment.GetEnvironmentVariable("GITHUB_TOKEN");
        if (!string.IsNullOrEmpty(token))
        {
            _http.DefaultRequestHeaders.Authorization = 
                new AuthenticationHeaderValue("Bearer", token);
        }
    }

    public async Task<string> FetchAsync(string sourceUrl, string productUrl, CancellationToken ct)
    {
        var apiUrl = GitHubUrlHelper.ConvertToApiUrl(sourceUrl, productUrl);
        
        Console.WriteLine($"[GitHub] Fetching: {apiUrl}");
        
        var response = await _http.GetAsync(apiUrl, ct);
        
        if (!response.IsSuccessStatusCode)
        {
            var err = await response.Content.ReadAsStringAsync(ct);
            Console.WriteLine($"[GitHub] Error {(int)response.StatusCode}: {err}");
            
            if (response.StatusCode == System.Net.HttpStatusCode.Forbidden)
            {
                Console.WriteLine("[GitHub] Rate limit exceeded. Consider adding GITHUB_TOKEN");
            }
            
            response.EnsureSuccessStatusCode();
        }
        
        return await response.Content.ReadAsStringAsync(ct);
    }
}