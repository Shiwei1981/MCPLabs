# MCP Servers and Clients Lab Codes  
  
Here I'll provide my lab codes for **MCP Servers** and **MCP Clients**. Most of the codes will be in Python.  
  
## Implementation Details  
  
- I build and invoke MCP Server codes on Python coded MCP Clients.  
- There will be **no document** to show how to invoke MCP Server in Cline, Claude or other tools.  

# MCP Client and Server Documentation  
  
## MCPClient.py  
**English:**   
MCPClient.py establishes a Python MCP Client, facilitating integration with your custom-developed AI Agent code. It is a simple modification based on the official sample code. For the Large Language Model (LLM), I use the GPT-4o mini model from Azure OpenAI. The library for LLM is AzureOpenAI by OpenAI, which allows direct function calls. To connect the MCP Client to a Server, simply use `StdioServerParameters("Python" + file path)`.  
**中文:**   
MCPClient.py 建立了一个 Python 的 MCP Client，方便与自己开发的 AI Agent 代码集成。这是基于官方样例代码的简单修改。LLM 我使用的是 Azure OpenAI 的 GPT-4o mini 模型。LLM 的库是 OpenAI 的 AzureOpenAI 库，支持直接使用 Function call 能力。MCP Client 连接 Server 的方式，只需使用 `StdioServerParameters("Python" + 文件路径)` 即可。  
  
## MCPServerCSVQuery.py  
**English:**   
MCPServerCSVQuery.py is a Python MCP Server that reads CSV files into memory, stores them in SQLITE, and performs queries. I modified this code after frequent errors from the original version I downloaded from unravel-team/mcp-analyst. The use of Polars, which are not necessary for data computation, is pending removal. After integration with function calls, the LLM first retrieves a list of files, then acquires metadata, followed by generating metadata-based query statements that are highly effective.  
**中文:**   
MCPServerCSVQuery.py 是一个 Python 的 MCP Server，它将 CSV 文件读入内存，存储在 SQLITE 中，并进行查询。这段代码是我在下载 unravel-team/mcp-analyst 的代码后，因频繁出现错误而进行的修改。里面的 Polars 并不用于数据计算，我还未将其删除。与 Function call 整合后，LLM 首先会获取文件列表，然后获取元数据，随后生成基于元数据的查询语句，这些查询语句的成功率非常高。  
  
## Future Developments  
**English:**   
I will soon update the MCP Client to connect to multiple MCP Servers.  
**中文:**   
我很快会更新 MCP Client，使其能连接多个 MCP Server。  
  
## Additional Information  
**English:**   
Currently, only simple comments have been written. The program itself is runnable. If there are any issues, please contact me directly. Thank you.  
**中文:**   
目前只写了一些简单的注释。程序本身是可以运行的。如果有问题，请直接联系我。多谢。  
