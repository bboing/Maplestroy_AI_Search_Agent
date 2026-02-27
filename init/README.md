# 초기 데이터 디렉토리

이 디렉토리는 데이터베이스 초기화 파일을 저장합니다.

## 파일 구조

```
init/
├── 01_maple_data.sql        # PostgreSQL 초기 데이터 (자동 생성)
└── README.md                 # 이 파일
```

## 사용 방법

### 1. 데이터 Export (데이터 소유자)

현재 로컬 PostgreSQL 데이터를 SQL 파일로 export:

```bash
python scripts/export_postgres_data.py
```

생성된 `01_maple_data.sql` 파일을 Git에 커밋하여 팀원과 공유합니다.

### 2. 데이터 Import (신규 사용자)

초기화 스크립트 실행 시 자동으로 import됩니다:

```bash
python scripts/init_all_databases.py
```

또는 수동으로:

```bash
psql -h localhost -U admin -d maple < init/01_maple_data.sql
```

## 주의사항

- `*.sql` 파일은 `.gitignore`에 포함되어 있어 자동으로 Git에 추적되지 않습니다
- 공유하려면 명시적으로 `git add -f init/01_maple_data.sql`로 추가하거나
- `.gitignore`를 수정하여 `!init/01_maple_data.sql`로 예외 처리해야 합니다

## 업데이트 주기

데이터가 변경될 때마다 export 후 커밋하여 최신 상태를 유지합니다.
