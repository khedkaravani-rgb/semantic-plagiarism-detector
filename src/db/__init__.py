from .auth import ( 
    add_user,
    delete_user,
    disable_2fa,
    enable_2fa,
    get_2fa_status,
    get_all_users,
    get_user_role,
    init_db,
    update_password,
    verify_user,
)
from .corpus_db import (
    add_chunks,
    add_document,
    clear_all_data,
    delete_document,
    get_all_documents,
    get_all_embeddings,
    get_chunk_registry,
    get_document_by_hash,
    get_document_chunks_count,
    get_documents_by_class,
    get_unique_class_sections,
    init_corpus_db,
)

__all__ = [
    "init_db",
    "verify_user",
    "get_user_role",
    "get_all_users",
    "add_user",
    "delete_user",
    "update_password",
    "get_2fa_status",
    "enable_2fa",
    "disable_2fa",
    "init_corpus_db",
    "add_document",
    "get_document_by_hash",
    "get_all_documents",
    "add_chunks",
    "get_chunk_registry",
    "get_all_embeddings",
    "delete_document",
    "clear_all_data",
    "get_document_chunks_count",
    "get_unique_class_sections",
    "get_documents_by_class",
]

 
from .migrations import (    # noqa: F401
    AUTH_SCHEMA_VERSION as AUTH_SCHEMA_VERSION,
    CORPUS_SCHEMA_VERSION as CORPUS_SCHEMA_VERSION,
    column_exists as column_exists,
    get_user_version as get_user_version,
    index_exists as index_exists,
    migrate_auth_database as migrate_auth_database,
    migrate_corpus_database as migrate_corpus_database,
    table_exists as table_exists,
)

__all__.extend(
    [
        "AUTH_SCHEMA_VERSION",
        "CORPUS_SCHEMA_VERSION",
        "column_exists",
        "get_user_version",
        "index_exists",
        "migrate_auth_database",
        "migrate_corpus_database",
        "table_exists",
    ]
)
