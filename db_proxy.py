from typing import Dict, List, Tuple, Any
import threading
import sqlite3
from guardrails import SQLGuard

class DBProxy:
    def __init__(self, dsn: str, security_conf: Dict, dialect: str = "sqlite"):
        self.dsn = dsn
        self.guard = SQLGuard(security_conf)
        self.dialect = dialect

    def _connect(self):
        return sqlite3.connect(self.dsn)

    def execute_safe(self, sql: str, params: Dict = None) -> Tuple[List[str], List[Tuple[Any,...]]]:
        checked = self.guard.validate_and_rewrite(sql, dialect=self.dialect)
        safe_sql = checked["sql"]
        timeout_ms = checked["timeout_ms"]
        self.guard.require_bound_params(safe_sql, params or {})

        conn = self._connect()
        cur = conn.cursor()

        result_columns: List[str] = []
        result_rows: List[Tuple[Any,...]] = []
        err = []

        def run():
            try:
                if params:
                    cur.execute(safe_sql, params)
                else:
                    cur.execute(safe_sql)
                if cur.description:
                    result_columns.extend([d[0] for d in cur.description])
                    result_rows.extend(cur.fetchall())
            except Exception as e:
                err.append(e)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout_ms / 1000.0)

        if t.is_alive():
            try:
                cur.close(); conn.close()
            except:
                pass
            raise TimeoutError(f"Query timed out after {timeout_ms} ms")

        if err:
            raise err[0]

        max_rows = self.guard.max_rows
        if len(result_rows) > max_rows:
            result_rows = result_rows[:max_rows]

        cur.close(); conn.close()
        return result_columns, result_rows
