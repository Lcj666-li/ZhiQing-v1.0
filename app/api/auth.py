"""简单 JWT 鉴权：登录签发 token，业务接口用 Bearer token 保护。

单账号（config.ini [auth] 段配置用户名/密码），不做用户表/注册。
REST 通过 Depends(require_auth) 校验 Authorization: Bearer <token>；
WebSocket 无法自定义请求头，token 走 ?token=，握手用同一个 verify_token 校验，失败 close(4401)。
token 自包含（sub + exp），服务端不存会话状态。
生产环境务必通过环境变量 JWT_SECRET 覆盖默认密钥。
"""

from __future__ import annotations

import time

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.conf.settings import settings

# auto_error=False：缺 Authorization 头时不自动抛 403，由我们自己统一回 401
_bearer = HTTPBearer(auto_error=False)


def create_token(username: str) -> str:
    """签发 JWT：sub=用户名，exp=当前时间 + 配置的有效期（分钟）。"""
    payload = {
        "sub": username,
        "exp": int(time.time()) + settings.JWT_EXPIRE_MINUTES * 60,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> str:
    """校验 JWT（签名 + 过期时间），返回用户名；无效/过期抛 ValueError。

    REST 依赖与 WebSocket 握手共用这一个校验函数，保证两条通道口径一致。
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise ValueError(f"无效或过期的 token：{exc}") from exc
    return str(payload.get("sub", ""))


def require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    """FastAPI 依赖：校验 Bearer token，返回用户名；缺失/无效统一 401。"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="缺少 Authorization: Bearer <token>",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        return verify_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc),
                            headers={"WWW-Authenticate": "Bearer"}) from exc