# exec_utils.py
# -----------------------------------------------------------------------------
# 목적:
#  - 생성된 SQL을 "세이프 미리보기"로 실행하고
#  - 실패 시 오류 유형/메시지를 바탕으로 LLM으로 자동 수정(Refine)하여
#  - 최대 N회 재시도 후 최종 SQL과 실행 리포트를 반환합니다.
#
# 전제:
#  - SQLite DB 파일을 사용 (Chinook_Sqlite.sqlite 등)
#  - LLM Refiner는 text2sql_agent.py의 refiner_agent를 사용
#
# 사용 예:
# from exec_utils import try_execute_with_refine
# final_sql, report = try_execute_with_refine("Chinook_Sqlite.sqlite", question, {"dialect":"sqlite"}, draft_sql)
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import sqlite3
import time
import re
import json

# =========================
# 결과 리포트 타입
# =========================

@dataclass
class ExecReport:
    ok: bool
    rows_preview: Optional[List[Dict[str, Any]]] = None
    row_count: int = 0
    err_type: Optional[str] = None     # "SYNTAX" | "SCHEMA" | "SEMANTIC" | "TIMEOUT" | "RUNTIME" | "SAFETY" | "UNKNOWN"
    err_message: Optional[str] = None
    # 필요하면 plan_hint, duration_ms, etc. 추가 가능


# =========================
# 세이프가드 / 유틸
# =========================

_DML_DDL_PATTERN = re.compile(r"^\s*(insert|update|delete|merge|create|alter|drop|truncate)\b", re.I)

def _attach_preview_limit(sql: str, dialect: str = "sqlite", limit: int = 50) -> str:
    """
    결과 미리보기 제한을 강제 삽입합니다.
    - sqlite/mysql/postgres: LIMIT N
    - oracle: FETCH FIRST N ROWS ONLY
    - mssql: (간단화) 현재는 건드리지 않음
    """
    s = sql.strip().rstrip(";")
    low = s.lower()

    # 이미 LIMIT/FETCH/TOP이 있으면 그대로 둠
    if (" limit " in low) or (" fetch first " in low) or low.startswith("select top "):
        return s + ";"

    if dialect in ("sqlite", "mysql", "postgresql", "postgres"):
        s = f"{s} LIMIT {limit}"
    elif dialect in ("oracle",):
        s = f"{s} FETCH FIRST {limit} ROWS ONLY"
    else:
        # 미지원 다이얼렉트: 안전하게 그대로 반환 (필요 시 추가)
        pass

    return s + ";"


def wrap_safe(sql: str, dialect: str = "sqlite", limit: int = 50) -> str:
    """
    - DML/DDL 금지
    - SELECT 전용
    - 미리보기 LIMIT 강제
    """
    if not sql or not sql.strip():
        raise ValueError("빈 SQL 입니다.")
    head = sql.strip()
    if _DML_DDL_PATTERN.match(head):
        raise ValueError("DML/DDL 금지. SELECT-only 모드입니다.")
    # SELECT 시작 여부는 너무 엄격히 보지 않되, 최소 방어
    if not head.lower().startswith("select "):
        # WITH로 시작하는 CTE도 허용: with ... select ...
        if not head.lower().startswith("with "):
            raise ValueError("SELECT 또는 WITH로 시작하는 읽기 전용 쿼리만 허용됩니다.")

    return _attach_preview_limit(sql, dialect=dialect, limit=limit)


# =========================
# 실행 (SQLite)
# =========================

def execute_preview_sqlite(db_path: str, sql: str, limit: int = 50) -> ExecReport:
    """
    SQLite에서 미리보기(최대 limit행)로 쿼리를 실행하고 결과를 반환합니다.
    - 자동 LIMIT 삽입
    - DML/DDL 차단
    - 기본 예외 유형 매핑
    """
    try:
        safe_sql = wrap_safe(sql, dialect="sqlite", limit=limit)
    except Exception as e:
        return ExecReport(ok=False, err_type="SAFETY", err_message=str(e))

    con = None
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cur = con.execute(safe_sql)
        rows = [dict(r) for r in cur.fetchmany(limit)]
        return ExecReport(ok=True, rows_preview=rows, row_count=len(rows))
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "no such table" in msg or "no such column" in msg:
            err_t = "SCHEMA"
        elif "syntax error" in msg:
            err_t = "SYNTAX"
        else:
            err_t = "RUNTIME"
        return ExecReport(ok=False, err_type=err_t, err_message=str(e))
    except sqlite3.DatabaseError as e:
        return ExecReport(ok=False, err_type="RUNTIME", err_message=str(e))
    except Exception as e:
        return ExecReport(ok=False, err_type="UNKNOWN", err_message=str(e))
    finally:
        try:
            if con:
                con.close()
        except:
            pass


# =========================
# LLM Refine 연결
# =========================

def _extract_sql_from_refiner_payload(payload: str) -> str:
    """
    refiner_agent의 응답(content 문자열)에서 SQL을 뽑아냅니다.
    기대 JSON:
        {"refined_sql": "...", "changes": [...], "confidence": 0.x}
    - 코드블록이 섞여 들어오는 경우를 대비하여 방어
    """
    s = payload.strip()
    # ```json ... ``` 제거
    if "```" in s:
        parts = s.split("```")
        # 가장 긴 JSON스러워 보이는 부분을 찾기 (대략적)
        cands = [p for p in parts if "{" in p and "}" in p]
        if cands:
            s = cands[-1]

    data = json.loads(s)  # 실패 시 예외 발생 → 상위에서 처리
    refined = data.get("refined_sql")
    if not refined:
        # 혹시 키가 다른 경우 대비
        for k in data.keys():
            if "sql" in k.lower():
                refined = data[k]
                break
    if not refined or not str(refined).strip():
        raise ValueError("Refiner 응답에 refined_sql이 없습니다.")
    return str(refined).strip()


def refine_with_llm(question: str, db_desc: Dict[str, Any], prev_sql: str,
                    err_type: str, err_message: str) -> str:
    """
    LLM Refiner 호출: text2sql_agent.refiner_agent를 재사용합니다.
    - 입력: 이전 SQL과 오류 타입/메시지
    - 출력: 보정된 SQL 문자열
    """
    # 순환 import 방지: 함수 내부에서 import
    from text2sql_agent import refiner_agent
    from langchain_core.messages import HumanMessage

    msg = (
        "다음 SQL을 개선하세요.\n"
        f"sql: {prev_sql}\n"
        f"question: {question}\n"
        f"error_type: {err_type}\n"
        f"error_message: {err_message}"
    )
    out = refiner_agent.invoke({"messages": [HumanMessage(content=msg)]})
    content = out["messages"][-1].content
    try:
        refined_sql = _extract_sql_from_refiner_payload(content)
    except Exception as e:
        # Refiner가 JSON 형태를 안 지킨 경우에 대한 마지막 방어
        # content 전체에서 간이로 SELECT ... 추출
        refined_sql = _best_effort_sql_extract(content)
        if not refined_sql:
            raise RuntimeError(f"Refiner 응답 파싱 실패: {e}\nRAW: {content}")
    return refined_sql


_SQL_BLOCK = re.compile(r"select\s.+", re.I | re.S)

def _best_effort_sql_extract(text: str) -> Optional[str]:
    """
    최악의 경우를 대비한 비상 추출기(설명 섞여도 select로 시작하는 블록을 긁어온다).
    *운영에서는 가급적 사용하지 않는 것이 좋음.*
    """
    m = _SQL_BLOCK.search(text or "")
    if not m:
        return None
    cand = m.group(0).strip()
    # 뒤쪽에 설명이 붙는 경우 세미콜론 기준으로 슬라이스
    if ";" in cand:
        cand = cand.split(";")[0]
    return cand.strip()


# =========================
# 메인 루프
# =========================

def try_execute_with_refine(db_path: str,
                            question: str,
                            db_desc: Dict[str, Any],
                            sql: str,
                            max_try: int = 3,
                            preview_limit: int = 50,
                            sleep_base_sec: float = 0.3) -> Tuple[str, ExecReport]:
    """
    1) SQL 미리보기 실행
    2) 실패 시 오류 타입/메시지로 Refine
    3) 최대 max_try 회 반복
    반환: (최종 SQL, 마지막 실행 리포트)
    """
    attempts = 0
    last_report: Optional[ExecReport] = None

    while attempts < max_try:
        report = execute_preview_sqlite(db_path, sql, limit=preview_limit)
        if report.ok:
            return sql, report

        # 실패 → LLM로 수정
        sql = refine_with_llm(
            question=question,
            db_desc=db_desc or {},
            prev_sql=sql,
            err_type=report.err_type or "",
            err_message=report.err_message or ""
        )
        attempts += 1
        last_report = report
        # 지수 백오프(최대 2초)
        time.sleep(min(sleep_base_sec * (2 ** attempts), 2.0))

    # 재시도 초과: 마지막 시도에서의 보정 SQL을 반환
    if last_report is None:
        last_report = ExecReport(ok=False, err_type="UNKNOWN", err_message="No attempts executed")
    return sql, last_report