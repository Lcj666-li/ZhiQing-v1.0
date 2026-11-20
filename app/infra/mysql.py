"""MySQL 访问层：FAQ 知识表 + 会话表的基础操作。

不含业务判断（业务编排在 services 层），上层通过 MySqlRepo 仓储对象访问 MySQL。

【设计要点】
    · INSERT IGNORE + 唯一索引 => 幂等，重复导入不产生脏数据；
    · recent_turns 按自增 id 倒排取 N 轮再反转 => 避免时间戳并发相同时排序不稳；
    · 软删除 is_deleted => "清空会话"只打标记，数据资产保留可审计；
    · 联合索引 idx_session(session_id, created_at) => 按会话拉历史走索引。
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, Sequence

import pymysql

from app.conf.settings import settings
from app.conf.logging import logger


class MySqlRepo:
    """轻量 MySQL 仓储。使用 pymysql 长连接 + 单一游标。

    设计取舍：不引入连接池 / ORM，直接持有一条长连接和一个游标。
    对本项目这种"低并发、以查询为主"的知识库场景足够，代码也更好读。
    注意：autocommit=False，写操作需要显式 commit（见下方各写方法）。
    """

    def __init__(self, database: str | None = None, create_database: bool = True) -> None:
        """建立到 MySQL 的连接。

        database：目标库名，None 用配置默认；create_database：True 时先尝试自动建库（需 CREATE 权限）。
        """
        self.database = database or settings.MYSQL_DATABASE
        if create_database:
            self._ensure_database()
        try:
            self.connect = pymysql.connect(
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                database=self.database,
                # utf8mb4：完整 4 字节 UTF-8，支持中文与 emoji
                charset="utf8mb4",
                # 关闭自动提交：把"提交时机"交给业务显式控制，便于批量导入时攒够再 commit
                autocommit=False,
            )
            self.cursor = self.connect.cursor()
            logger.info("MySQL 连接成功：%s:%s/%s", settings.MYSQL_HOST, settings.MYSQL_PORT, self.database)
        except Exception as exc:  # pragma: no cover - 依赖外部服务
            logger.error("MySQL 连接失败：%s", exc)
            raise

    # ------------------------------------------------------------------ #
    # 连接与建库
    # ------------------------------------------------------------------ #
    def _ensure_database(self) -> None:
        """库不存在时自动创建（需要账号有 CREATE 权限）。

        用 CREATE DATABASE IF NOT EXISTS 保证幂等：库已存在也不会报错。
        这里单独开一条"不带 database"的临时连接来建库，建完即关；
        若账号无权限则只告警不抛出（生产环境库通常已由 DBA 建好）。
        """
        try:
            conn = pymysql.connect(
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                charset="utf8mb4",
            )
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.database}` "
                    f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            conn.commit()
            conn.close()
        except Exception as exc:  # pragma: no cover
            logger.warning("自动建库跳过（可能无权限）：%s", exc)

    def execute(self, sql: str, params: Sequence | None = None):
        """执行一条 SQL，返回游标（调用方可继续 fetchall / 读 rowcount）。

        params 用占位符 %s 传参（参数化查询），由 pymysql 负责转义，防止 SQL 注入。
        注意：这里不 commit，写操作是否落盘由调用方决定。
        """
        self.cursor.execute(sql, params)
        return self.cursor

    def commit(self) -> None:
        self.connect.commit()

    def rollback(self) -> None:
        """回滚当前事务；回滚本身失败也只静默忽略。"""
        try:
            self.connect.rollback()
        except Exception:  # pragma: no cover
            pass

    def close(self) -> None:
        """关闭游标与连接；关闭异常只告警，不影响主流程退出。"""
        try:
            self.cursor.close()
            self.connect.close()
            logger.info("MySQL 连接已关闭")
        except Exception as exc:  # pragma: no cover
            logger.warning("MySQL 关闭异常：%s", exc)

    # ------------------------------------------------------------------ #
    # FAQ 知识表
    # ------------------------------------------------------------------ #
    def ensure_faq_table(self) -> None:
        """建 FAQ 表（不存在才建）。

        question(191) 前缀唯一索引：保证同一问句只存一条（配合 INSERT IGNORE 幂等）；
        191 是 utf8mb4 下 InnoDB 单列索引的安全长度上限。
        """
        sql = f"""
            CREATE TABLE IF NOT EXISTS `{settings.FAQ_TABLE}` (
                id          BIGINT AUTO_INCREMENT PRIMARY KEY,
                subject     VARCHAR(64)  NOT NULL DEFAULT '' COMMENT '学科/来源',
                question    VARCHAR(512) NOT NULL COMMENT '问题',
                answer      MEDIUMTEXT   NOT NULL COMMENT '答案',
                created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_question (question(191))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
              COMMENT='FAQ 标准问句与答案'
        """
        self.execute(sql)
        self.commit()
        logger.info("FAQ 表就绪：%s", settings.FAQ_TABLE)

    def import_faq_csv(self, csv_file: str | None = None) -> tuple[int, int]:
        """导入 FAQ CSV（列：学科名称/问题/答案），返回 (新增数, 跳过数)。

        幂等：已存在的问句被 INSERT IGNORE 跳过，反复导入不产生重复数据。
        每新增 100 条 commit 一次，避免长事务占用过多 undo/锁资源。
        """
        csv_file = csv_file or settings.FAQ_CSV
        if not os.path.exists(csv_file):
            logger.error("FAQ CSV 不存在：%s", csv_file)
            return 0, 0

        self.ensure_faq_table()
        inserted = skipped = 0
        with open(csv_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = (row.get("问题") or "").strip()
                answer = (row.get("答案") or "").strip()
                subject = (row.get("学科名称") or "").strip()
                if not question or not answer:
                    skipped += 1
                    continue
                # INSERT IGNORE + 唯一索引 => 天然幂等
                affected = self.execute(
                    f"INSERT IGNORE INTO `{settings.FAQ_TABLE}` (subject, question, answer) VALUES (%s, %s, %s)",
                    (subject, question, answer),
                ).rowcount
                if affected:
                    inserted += 1
                else:
                    skipped += 1
                # 每新增 100 条提交一次：攒批减少 commit 次数，又不至于事务过大
                if inserted % 100 == 0:
                    self.commit()
        self.commit()
        logger.info("FAQ 导入完成：新增 %s 条，跳过 %s 条（文件 %s）", inserted, skipped, csv_file)
        return inserted, skipped

    def all_questions(self) -> list[str]:
        """取出全部标准问句（按 id 升序），供 FAQ 通道向量化入库（scripts/init_faq_store.py）。

        读失败只记日志并返回空列表：FAQ 通道宁可"查不到"也不让异常冒泡打断服务。
        """
        try:
            self.execute(f"SELECT question FROM `{settings.FAQ_TABLE}` ORDER BY id")
            return [row[0] for row in self.cursor.fetchall()]
        except Exception as exc:
            logger.error("读取 FAQ 问题失败：%s", exc)
            return []

    def answer_of(self, question: str) -> str | None:
        """按标准问句精确回查答案（向量命中问句后用它取答案）。找不到返回 None。"""
        try:
            self.execute(f"SELECT answer FROM `{settings.FAQ_TABLE}` WHERE question = %s LIMIT 1", (question,))
            row = self.cursor.fetchone()
            return row[0] if row else None
        except Exception as exc:
            logger.error("读取 FAQ 答案失败：%s", exc)
            return None

    def count_faq(self) -> int:
        """统计 FAQ 条数（用于健康检查/导入结果核对）。"""
        self.execute(f"SELECT COUNT(*) FROM `{settings.FAQ_TABLE}`")
        return int(self.cursor.fetchone()[0])

    # ------------------------------------------------------------------ #
    # 会话表
    # ------------------------------------------------------------------ #
    def ensure_chat_table(self) -> None:
        """建多轮会话表（不存在才建）。

        is_deleted 软删除标记："清空会话"只置 1，不物理删除，数据资产保留；
        idx_session(session_id, created_at) 联合索引让按会话拉历史直接走索引。
        """
        sql = f"""
            CREATE TABLE IF NOT EXISTS `{settings.CHAT_TABLE}` (
                id          BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id  VARCHAR(64) NOT NULL COMMENT '会话 ID',
                question    TEXT        NOT NULL,
                answer      MEDIUMTEXT  NOT NULL,
                created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
                is_deleted  TINYINT(1)  NOT NULL DEFAULT 0,
                KEY idx_session (session_id, created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
              COMMENT='多轮会话记录'
        """
        self.execute(sql)
        self.commit()
        logger.info("会话表就绪：%s", settings.CHAT_TABLE)

    def insert_turn(self, session_id: str, question: str, answer: str) -> None:
        """落库一轮问答（question + answer），立即 commit。"""
        self.execute(
            f"INSERT INTO `{settings.CHAT_TABLE}` (session_id, question, answer) VALUES (%s, %s, %s)",
            (session_id, question, answer),
        )
        self.commit()

    def recent_turns(self, session_id: str, limit: int = 5) -> list[dict]:
        """取最近 N 轮对话，返回时间正序（旧 -> 新）。

        为什么"按 id 倒排取 N 条，再反转"，而不是直接按 created_at 排序？
            · 自增 id 严格单调递增，天然反映插入先后；
            · created_at 精度只到秒，高并发下同一秒可能写入多轮，时间戳相同会导致
              ORDER BY created_at 的相对顺序不稳定（数据库不保证同值行的次序）；
            · 用 id DESC LIMIT N 先锁定"最新的 N 条"，再 reversed 还原成"旧->新"，
              正好是喂给大模型做上下文时期望的阅读顺序。
        只取 is_deleted=0 的行：软删除的历史不再参与上下文。
        读失败返回空列表（多轮上下文缺失不致命，退化为单轮问答）。
        """
        try:
            self.execute(
                f"SELECT question, answer FROM `{settings.CHAT_TABLE}` "
                f"WHERE session_id = %s AND is_deleted = 0 ORDER BY id DESC LIMIT %s",
                (session_id, limit),
            )
            rows = [{"question": q, "answer": a} for q, a in self.cursor.fetchall()]
            # 倒排取到的是"新->旧"，反转成"旧->新"再交给上层拼上下文
            return list(reversed(rows))
        except Exception as exc:
            logger.error("读取会话历史失败：%s", exc)
            return []

    def soft_delete_turns(self, session_id: str) -> bool:
        """软删除：保留数据资产，只打删除标记。

        把该会话所有行的 is_deleted 置 1（而非 DELETE），用户侧"清空会话"即时生效，
        但底层数据仍可审计、可回溯、可做离线分析。成功返回 True，失败回滚并返回 False。
        """
        try:
            self.execute(
                f"UPDATE `{settings.CHAT_TABLE}` SET is_deleted = 1 WHERE session_id = %s", (session_id,)
            )
            self.commit()
            return True
        except Exception as exc:
            logger.error("清除会话失败：%s", exc)
            self.rollback()
            return False
