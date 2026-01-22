from typing import Dict
import re
import sqlglot
from sqlglot import exp

class SQLGuard:
    def __init__(self, security_conf: Dict):
        self.allowed_schemas = set([s.upper() for s in security_conf.get("allowed_schemas", [])])
        self.allowed_statements = set(security_conf.get("allowed_statements", ["SELECT"]))
        self.max_rows = int(security_conf.get("max_rows", 50000))
        self.default_timeout_ms = int(security_conf.get("default_timeout_ms", 15000))
        self.forbidden_keywords = set(security_conf.get("forbidden_keywords", []))

    def _contains_forbidden_keywords(self, sql: str):
        upper = sql.upper()
        for kw in self.forbidden_keywords:
            if re.search(rf"\b{kw}\b", upper):
                raise ValueError(f"Forbidden keyword detected: {kw}")

    def _validate_ast(self, sql: str, dialect: str = "sqlite"):
        try:
            parsed = sqlglot.parse_one(sql, read=dialect)
        except Exception as e:
            raise ValueError(f"SQL parse error: {e}")

        if not isinstance(parsed, (exp.Select, exp.Union, exp.With)):
            raise ValueError("Only SELECT/CTE/UNION statements are allowed.")

        banned_nodes = (exp.Update, exp.Delete, exp.Insert, exp.Create, exp.Alter, exp.Drop, exp.Truncate, exp.Merge)
        if list(parsed.find_all(banned_nodes)):
            raise ValueError("DML/DDL statements are not allowed.")

        # sqlite에는 스키마가 약하므로 Oracle 전환 시 아래 로직을 활성화
        # for tbl in parsed.find_all(exp.Table):
        #     schema = tbl.args.get("db")
        #     if schema is None:
        #         raise ValueError(f"Schema is required for table: {tbl}")
        #     if schema.upper() not in self.allowed_schemas:
        #         raise ValueError(f"Schema not allowed: {schema}")

    def _enforce_limit(self, sql: str) -> str:
        up = sql.upper()
        if "LIMIT " in up or "FETCH FIRST " in up:
            return sql
        return sql.rstrip() + f"\nLIMIT {self.max_rows}"

    def validate_and_rewrite(self, sql: str, dialect: str = "sqlite") -> Dict:
        self._contains_forbidden_keywords(sql)
        self._validate_ast(sql, dialect=dialect)
        safe_sql = self._enforce_limit(sql)
        return {"sql": safe_sql, "timeout_ms": self.default_timeout_ms, "max_rows": self.max_rows}

    def require_bound_params(self, sql: str, params: Dict):
        # 문자열 하드코딩 과다 사용 방지(간단 가드)
        if re.search(r"'[^']{20,}'", sql):
            raise ValueError("Large string literal detected. Use bound parameters.")
