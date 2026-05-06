FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

COPY examples/xdt_demo/xdt_api/xdt_api.csproj ./xdt_api/
COPY examples/xdt_demo/xdt_mgr/xdt_mgr.csproj ./xdt_mgr/
RUN dotnet restore ./xdt_mgr/xdt_mgr.csproj

COPY examples/xdt_demo/xdt_api/ ./xdt_api/
COPY examples/xdt_demo/xdt_mgr/ ./xdt_mgr/

WORKDIR /src/xdt_mgr
RUN dotnet publish ./xdt_mgr.csproj -c Release -o /app

FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app

COPY --from=build /app .

COPY examples/xdt_demo/xdt_mgr/xdt_rpo.json ./xdt_rpo.json

ENTRYPOINT ["dotnet", "xdt_mgr.dll"]