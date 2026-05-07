namespace xdt_api.Models;

public class ProductConfig
{
    public string Name { get; set; } = string.Empty;
    public string ReleasesUrl { get; set; } = string.Empty;
}

public class SourceConfig
{
    public string Name { get; set; } = string.Empty;
    public string SourceUrl { get; set; } = string.Empty;
    public string FetchMethod { get; set; } = string.Empty;
    public List<ProductConfig> Products { get; set; } = new();
}

public class CollectorConfig
{
    public List<SourceConfig> Sources { get; set; } = new();
}