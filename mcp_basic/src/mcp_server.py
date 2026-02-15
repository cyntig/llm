from fastmcp import FastMCP

# 创建 MCP Server 实例
mcpServer = FastMCP("Weather Server")

# 定义工具
@mcpServer.tool(
    name="get_live_weather",
    description="获取城市实时天气情况"
)
def get_live_weather(city: str) -> str:
    """获取指定城市的实时天气"""
    return f"{city} 晴天"

if __name__ == "__main__":
    # 在终端运行此脚本
    mcpServer.run()