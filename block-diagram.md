graph TD

  %% ───────── Client ─────────
  subgraph CLIENT[Client]
    BROWSER[사용자 브라우저]
    MCP_CLIENT[MCP Client\n(Continue, VSCode 등)]
  end

  %% ───────── UI Layer ─────────
  subgraph UI[Presentation Layer\n(app.py)]
    CHAT[Chat UI & Session State\n(Streamlit Chat)]
    PREVIEW[SQL 미리보기 & 결과 테이블\n(execute_preview_sqlite)]
  end

  %% ───────── Text2SQL Core ─────────
  subgraph CORE[Text2SQL Core\n(text2sql_agent.py)]
    MR[Memory Retrieve Node\n(try_reuse_sql_from_memory)]
    SEL[Selector Agent\n(find_relevant_tables,\nget_table_columns)]
    DEC[Decomposer Agent]
    COM[Composer Agent]
    REF[Refiner Agent]
    MS[Memory Save Node]
  end

  %% ───────── MCP Server ─────────
  subgraph MCP[MCP Server Layer\n(server.py)]
    TOOL_SCHEMA[get_schema / get_relations\n(schema_tools 캐시 사용)]
    TOOL_GEN[generate_sql\n(make_query + try_execute_with_refine)]
    TOOL_RUN[run_sql_proxy\n(DBProxy.execute_safe)]
  end

  %% ───────── Infra / Data ─────────
  subgraph INFRA[Data & Security Layer]
    VS[Chroma Vector Store\n(sql_memory_chroma_db,\nAzureOpenAIEmbeddings)]
    SCHEMA[Schema & Hints\nChinook_Column_Config.json,\nrelations, join_hint,\nkeyword_map, relation_bonus_map]
    DB[(SQLite DB\nChinook_Sqlite.sqlite)]
    SEC[DBProxy & security.yaml]
  end

  %% ───────── Edges: Client → UI / MCP ─────────
  BROWSER --> CHAT
  CHAT -->|질문 (자연어)| CORE
  PREVIEW -->|SQL 실행| DB

  MCP_CLIENT -->|MCP Protocol (stdio)| MCP

  %% ───────── Edges: UI ↔ Core ─────────
  CHAT -->|make_query(AgentState)| CORE
  CORE -->|refined_sql 반환| CHAT
  CHAT --> PREVIEW

  %% ───────── Edges: Core 내부 플로우 ─────────
  MR -->|재사용 실패| SEL
  MR -->|재사용 성공 (sql)| REF
  SEL --> DEC --> COM --> REF --> MS

  %% ───────── Edges: Core ↔ Vector Store / Schema ─────────
  MR --> VS
  MS --> VS

  SEL --> SCHEMA
  COM --> SCHEMA

  %% ───────── Edges: MCP ↔ Core / DB / Schema ─────────
  MCP_CLIENT -.->|tool:get_schema|get_schema
  TOOL_SCHEMA --> SCHEMA

  MCP_CLIENT -.->|tool:generate_sql| TOOL_GEN
  TOOL_GEN --> CORE
  TOOL_GEN -->|try_execute_with_refine| DB

  MCP_CLIENT -.->|tool:run_sql_proxy| TOOL_RUN
  TOOL_RUN --> SEC --> DB