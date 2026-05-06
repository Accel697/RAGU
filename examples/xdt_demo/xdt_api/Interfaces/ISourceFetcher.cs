namespace xdt_api.Interfaces;

public interface ISourceFetcher
{
    Task<string> FetchAsync(string sourceUrl, string productUrl, CancellationToken ct = default);
}