namespace xdt_api.Sources.GitHub;

public static class GitHubUrlHelper
{
    private const string ApiBase = "https://api.github.com";

    public static string ConvertToApiUrl(string sourceUrl, string productUrl)
    {
        try
        {
            var cleanUrl = productUrl
                .Replace("https://github.com/", "", StringComparison.OrdinalIgnoreCase)
                .TrimEnd('/');

            var parts = cleanUrl.Split('/', StringSplitOptions.RemoveEmptyEntries);

            if (parts.Length < 2)
                throw new ArgumentException($"Invalid GitHub URL format: {productUrl}");

            var owner = parts[0];
            var repo = parts[1];

            var endpoint = productUrl.Contains("/tags", StringComparison.OrdinalIgnoreCase) 
                ? "tags" 
                : "releases";

            var apiUrl = $"{ApiBase}/repos/{owner}/{repo}/{endpoint}?per_page=30";

            return apiUrl;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GitHub] URL Conversion FAILED: {ex.Message}");
            Console.WriteLine($"[GitHub] Source URL: {sourceUrl}");
            Console.WriteLine($"[GitHub] Product URL: {productUrl}");
            throw;
        }
    }
}