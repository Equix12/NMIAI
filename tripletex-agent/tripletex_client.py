"""Tripletex API client wrapper with full request/response logging."""

import json
import httpx
import logging

logger = logging.getLogger("tripletex.api")


class TripletexClient:
    """Thin wrapper around the Tripletex v2 REST API."""

    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self.client = httpx.Client(timeout=30.0)
        self._call_count = 0

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        self._call_count += 1
        call_num = self._call_count

        # Log full request
        log_parts = [f"API #{call_num} >>> {method} {url}"]
        if "params" in kwargs and kwargs["params"]:
            log_parts.append(f"  params: {json.dumps(kwargs['params'], ensure_ascii=False, default=str)}")
        if "json" in kwargs and kwargs["json"] is not None:
            body_str = json.dumps(kwargs["json"], ensure_ascii=False, default=str)
            log_parts.append(f"  body: {body_str}")
        if "files" in kwargs:
            for name, file_tuple in kwargs["files"].items():
                fname = file_tuple[0] if isinstance(file_tuple, tuple) else str(file_tuple)
                size = len(file_tuple[1]) if isinstance(file_tuple, tuple) and len(file_tuple) > 1 else 0
                log_parts.append(f"  file: {name}={fname} ({size} bytes)")
        logger.debug("\n".join(log_parts))

        resp = self.client.request(method, url, auth=self.auth, **kwargs)

        # Parse response
        try:
            data = resp.json()
        except Exception:
            data = {"status": resp.status_code, "message": resp.text}

        # Log full response
        if resp.status_code >= 400:
            # Error responses: log everything at WARNING
            status = data.get("status", resp.status_code)
            message = data.get("message", "Unknown error")
            validation = data.get("validationMessages") or []
            dev_msg = data.get("developerMessage", "")

            error_parts = [f"API #{call_num} <<< ERROR {status} on {method} {path}: {message}"]
            if dev_msg:
                error_parts.append(f"  Developer message: {dev_msg}")
            for v in validation:
                field = v.get("field", "unknown")
                msg = v.get("message", "")
                error_parts.append(f"  Validation: field='{field}' — {msg}")

            # Log the full raw response body for debugging
            raw_body = json.dumps(data, ensure_ascii=False, default=str)
            error_parts.append(f"  Full response: {raw_body}")

            error_summary = "\n".join(error_parts)
            logger.warning(error_summary)

            data["_error_summary"] = error_summary
            data["_is_error"] = True
        else:
            # Success responses: log at DEBUG with full body
            resp_str = json.dumps(data, ensure_ascii=False, default=str)
            # For large responses (list queries), truncate in console but keep full in file
            if len(resp_str) > 2000:
                logger.debug("API #{call_num} <<< {method} {path} -> {status} ({length} chars)"
                           .format(call_num=call_num, method=method, path=path,
                                   status=resp.status_code, length=len(resp_str)))
                logger.debug("API #%d <<< full response:\n%s", call_num, resp_str)
            else:
                logger.debug("API #%d <<< %s %s -> %d: %s",
                           call_num, method, path, resp.status_code, resp_str)

        return data

    def get(self, path: str, params: dict | None = None) -> dict:
        if params is None:
            params = {}
        # Ensure pagination defaults so we don't miss results
        params.setdefault("count", 1000)
        params.setdefault("from", 0)
        return self._request("GET", path, params=params)

    def post(self, path: str, json_body: dict | None = None, params: dict | None = None) -> dict:
        kwargs = {"json": json_body}
        if params:
            kwargs["params"] = params
        return self._request("POST", path, **kwargs)

    def put(self, path: str, json_body: dict | None = None) -> dict:
        return self._request("PUT", path, json=json_body)

    def delete(self, path: str) -> dict:
        return self._request("DELETE", path)

    # ── Convenience helpers ───────────────────────────────────────

    def list_employees(self, fields: str = "id,firstName,lastName,email") -> list:
        r = self.get("employee", {"fields": fields})
        return r.get("values", [])

    def create_employee(self, data: dict) -> dict:
        return self.post("employee", data)

    def list_customers(self, fields: str = "id,name") -> list:
        r = self.get("customer", {"fields": fields})
        return r.get("values", [])

    def create_customer(self, data: dict) -> dict:
        return self.post("customer", data)

    def create_product(self, data: dict) -> dict:
        return self.post("product", data)

    def create_order(self, data: dict) -> dict:
        return self.post("order", data)

    def create_invoice(self, data: dict) -> dict:
        return self.post("invoice", data)

    def create_project(self, data: dict) -> dict:
        return self.post("project", data)

    def create_travel_expense(self, data: dict) -> dict:
        return self.post("travelExpense", data)

    def create_department(self, data: dict) -> dict:
        return self.post("department", data)

    def list_departments(self) -> list:
        r = self.get("department", {"fields": "id,name,departmentNumber"})
        return r.get("values", [])

    def create_contact(self, data: dict) -> dict:
        return self.post("contact", data)

    def create_supplier(self, data: dict) -> dict:
        return self.post("supplier", data)

    def create_voucher(self, data: dict) -> dict:
        return self.post("ledger/voucher", data)

    def list_accounts(self, fields: str = "id,number,name") -> list:
        r = self.get("ledger/account", {"fields": fields})
        return r.get("values", [])

    def import_bank_statement(self, file_data: bytes, filename: str) -> dict:
        return self._request(
            "POST", "bank/statement/import",
            files={"file": (filename, file_data)},
        )
