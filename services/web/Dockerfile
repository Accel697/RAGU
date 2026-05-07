FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

COPY services/web/web.csproj ./web.csproj
RUN dotnet restore ./web.csproj

COPY services/web/Program.cs ./Program.cs
COPY services/web/wwwroot ./wwwroot

RUN dotnet publish ./web.csproj -c Release -o /app

FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app .
ENTRYPOINT ["dotnet", "web.dll"]