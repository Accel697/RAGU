namespace xdt_api.Models;

public class CollectionResult
{
    public string Source { get; set; } = string.Empty;
    public string Product { get; set; } = string.Empty;
    public string RawData { get; set; } = string.Empty;
    public DateTime CollectedAt { get; set; } = DateTime.UtcNow;
}