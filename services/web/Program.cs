using System.Net.Http.Json;

var builder = WebApplication.CreateBuilder(args);

var raguUrl = Environment.GetEnvironmentVariable("RAGU_API_URL");

builder.Services.AddHttpClient("RaguClient", client =>
{
    client.BaseAddress = new Uri(raguUrl);
    client.Timeout = TimeSpan.FromMinutes(10);
});

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapPost("/api/ask", async (HttpRequest request, IHttpClientFactory factory) =>
{
    try
    {
        var body = await request.ReadFromJsonAsync<AskRequest>();
        if (string.IsNullOrWhiteSpace(body?.question))
            return Results.BadRequest(new { error = "Вопрос не может быть пустым" });

        var client = factory.CreateClient("RaguClient");
        var response = await client.PostAsJsonAsync("/query", new { question = body.question });

        if (!response.IsSuccessStatusCode)
            return Results.StatusCode(502);

        var result = await response.Content.ReadFromJsonAsync<AskResponse>();
        return Results.Json(new { answer = result?.answer ?? "Ответ не получен" });
    }
    catch
    {
        return Results.StatusCode(500);
    }
});

app.Run("http://*:8080");

record AskRequest(string? question);
record AskResponse(string? answer);