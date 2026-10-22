"""Expected failure modes, mapped to proper HTTP status codes by the app (never a bare 500)."""


class AppError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
