FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

COPY examples/webchat_demo/webchat_demo.csproj ./webchat_demo.csproj
RUN dotnet restore ./webchat_demo.csproj

COPY examples/webchat_demo/Program.cs ./Program.cs
COPY examples/webchat_demo/wwwroot ./wwwroot

RUN dotnet publish ./webchat_demo.csproj -c Release -o /app

FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app .
ENTRYPOINT ["dotnet", "webchat_demo.dll"]