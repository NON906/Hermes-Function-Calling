#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp import MCPToolkit

class McpManager:
    json_mtime = 0.0
    mcp_tools = []
    tasks = []
    is_exit = False
    loaded_tasks = 0

    async def load_json(self):
        if os.path.isfile('settings/mcp_config.json') and self.json_mtime != os.path.getmtime('settings/mcp_config.json'):
            self.is_exit = True
            await asyncio.gather(*self.tasks)
            self.tasks = []
            self.mcp_tools = []
            self.is_exit = False
            self.json_mtime = os.path.getmtime('settings/mcp_config.json')
            with open('settings/mcp_config.json', mode='r') as f:
                mcp_dict_all = json.load(f)
            self.loaded_tasks = 0
            for target in mcp_dict_all['mcpServers'].values():
                self.tasks.append(asyncio.create_task(self.add_server(target)))
            while self.loaded_tasks < len(self.tasks) and not self.is_exit:
                await asyncio.sleep(1)
            return True
        return False

    async def add_server(self, target):
        server_args = ['/c', target['command'], *target['args']]
        server_params = StdioServerParameters( 
            command='cmd', 
            args=server_args, 
        )
        async with stdio_client(server_params) as (read, write): 
            async with ClientSession(read, write) as session: 
                toolkit = MCPToolkit(session=session) 
                await toolkit.initialize() 
                self.mcp_tools += toolkit.get_tools()
                self.loaded_tasks += 1
                while not self.is_exit:
                    await asyncio.sleep(0.1)