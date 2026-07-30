# Arachne C++ Core -- 설계 노트

이 문서는 `cpp/` 아래의 코드가 어떻게 구성되어 있는지 세 가지 수준에서
설명합니다: 상태를 담는 **데이터 구조(Data structure) 수준**, 각 모듈이
어떻게 배치되고 서로 의존하는지의 **코드(Code) 수준**, 그리고 핵심
구성요소들의 실제 동작을 다루는 **함수(Function) 수준**입니다.

본 문서는 설명용 문서일 뿐이며, 현재 코드 상태를 기술하는 것이지 새로운
설계를 제안하는 것이 아닙니다.

## 1. 데이터 구조 수준

### 값 타입 (`include/arachne/types.hpp`)

가장 하위 계층은 다른 모든 모듈이 공유하는, 작고 trivially-copyable한
값 타입들입니다:

- `VectorId`, `RegionId` -- 단순한 `std::uint64_t` 별칭입니다. 식별자는
  불투명(opaque)하며, 하위 계층 어디에서도 키(key) 이상의 의미로
  다루지 않습니다.
- `VectorView { const float* data; std::uint32_t dim; }` -- 벡터의 raw
  구성요소에 대한 non-owning view로, 호출 한 번의 생명주기 동안만
  유효합니다 (직접 구현한 `std::span<const float>`라 볼 수 있습니다).
  어떤 코드도 `VectorView`를 그것을 받은 호출이 끝난 뒤까지 보관하지
  않습니다.
- `Query { VectorView vector; std::uint32_t top_k; }`
- `Record { VectorId id; VectorView vector; }`
- `Neighbor { VectorId id; float distance; }`
- `SearchResult { std::vector<Neighbor> neighbors; bool served_gpu_only; }`
- `InsertResult { bool ok; }`, `DeleteResult { bool ok; }`

이 타입들은 모든 레이어 경계(`Engine` -> `Core` -> `IAdapter`/`IRegion`)를
가로질러 사용되는 유일한 타입들입니다.

### Region 관련 타입 (`include/arachne/adapter/region.hpp`)

- `ResidencyState` -- `Host | Pending | Resident`: Region의 authoritative한
  mutable 상태가 현재 어디에 있는지를 나타냅니다.
- `RegionFootprint { std::vector<RegionId> regions; }` -- 어떤 연산이
  건드렸거나, 범위(scope)로 지정된 Region들의 집합입니다.
- `LeaseHandle { RegionId region; std::uint64_t epoch; bool valid(); }` --
  GPU write lease입니다. `valid()`는 `epoch != 0`이며, 기본 생성된
  handle은 항상 invalid입니다. `Core`와 이를 발급한 `IRegion` 구현체
  외부에서는 불투명하게 다뤄집니다.
- `ModificationDelta { std::vector<std::byte> payload; }` -- lease epoch
  동안 Region 내부에서 무엇이 바뀌었는지를 나타내는 index별 인코딩입니다.
  `Core` 수준에서는 불투명하게 남겨둡니다.
- `ReconciliationReport { bool closed; RegionFootprint touched_neighbors; }`

### Adapter 관련 타입 (`include/arachne/adapter/index_adapter.hpp`)

- `ExecutionMode` -- `GpuOnly | Hybrid`.
- `TraverseRequest { Query query; ExecutionMode mode; RegionFootprint scope; }`
- `TraverseResult { SearchResult result; RegionFootprint touched; bool completed_within_scope; }`
- `ModifyOp` -- `Insert | Delete`.
- `ModifyRequest { ModifyOp op; Record record; VectorId target; ExecutionMode mode; RegionFootprint scope; LeaseHandle lease; }`
- `ModifyResult { bool ok; RegionFootprint touched; RegionFootprint modified; }`

### Core가 소유하는 타입 (`include/arachne/core/anchor_manager.hpp`)

- `Stitch { RegionId region; LeaseHandle lease; }` -- Anchor(`VectorId`)와
  그 Anchor가 현재 GPU write 권한을 갖고 있는 Region 사이의 연관관계입니다.
  `AnchorManager`는 이를 anchor id로 키를 잡아 저장합니다:
  `std::unordered_map<VectorId, std::vector<Stitch>>`, 일반
  `std::mutex`로 보호됩니다.

### RoutingCache 내부 데이터 구조 (`src/core/routing_cache_hnsw.cpp`)

`ASRoutingCacheHnsw::Instance`는 hnswlib의 `HierarchicalNSW<float>` 인덱스
하나를 감싸는 private, `.cpp` 전용 wrapper입니다:

```
Instance
  hnswlib::L2Space space_
  hnswlib::HierarchicalNSW<float> index_
  std::unordered_set<VectorId> live_ids_
  std::size_t tombstones_
```

`ASRoutingCacheHnsw` 자체는 이러한 `Instance`를 `std::unique_ptr<Instance>
active_` 하나로만 보유하며, `std::shared_mutex mutex_`로 보호하고, 여기에
백그라운드 compaction을 위한 상태(`std::atomic<bool> compacting_`,
`std::thread compaction_thread_`)를 더한 형태입니다. `ASRoutingCacheHnsw`에
"shadow"용 두 번째 멤버는 의도적으로 두지 않았습니다 -- shadow `Instance`는
`compactImpl()` 내부의 순수한 지역 변수일 뿐이며, swap이 일어나는 순간에만
`active_`가 됩니다.

## 2. 코드 수준 -- 모듈 배치와 의존관계

```
include/arachne/
  types.hpp                        (의존 없음)
  logging.hpp                      (의존 없음; ARACHNE_WITH_RAFT로 분기)
  adapter/
    region.hpp                     -> types.hpp
    index_adapter.hpp               -> region.hpp, types.hpp
  core/
    routing_cache.hpp               -> types.hpp
    routing_cache_hnsw.hpp          -> routing_cache.hpp
    anchor_manager.hpp               -> adapter/region.hpp, types.hpp
    core.hpp                         -> adapter/index_adapter.hpp, core/anchor_manager.hpp, core/routing_cache.hpp
  interface/
    index.hpp                         (types.hpp 외 의존 없음)
    engine.hpp                       -> adapter/index_adapter.hpp, core/core.hpp, core/routing_cache.hpp, interface/index.hpp

src/
  core/routing_cache_hnsw.cpp        -> routing_cache_hnsw.hpp, <hnswlib/hnswlib.h>
  core/anchor_manager.cpp            -> anchor_manager.hpp
  core/core.cpp                      -> core.hpp, logging.hpp
  interface/engine.cpp                -> engine.hpp
```

세 개의 아키텍처 레이어가 있으며, 의존 방향은 한 방향입니다 (상위가
하위에 의존하며, 그 반대는 없습니다):

1. **Interface** -- `Index`, `Engine`. `Index`는 애플리케이션 코드가
   의존하는 추상 진입점입니다 -- pure-virtual `search`/`insert`/`remove`
   뿐이며 그 이상은 없습니다 -- 한 레이어 아래에서 `RoutingCache`가
   쓰는 패턴과 동일합니다: `Core`가 어떤 구체적인 `RoutingCache`를
   받았는지에 대해 무관심할 수 있듯이, 애플리케이션 코드가 실제로
   요청을 처리하는 구체적인 최상위 구현체가 무엇인지에 대해 무관심할
   수 있도록 존재합니다. `Engine`은 첫 번째(이자 현재로서는 유일한)
   `Index` 구현체입니다. `IAdapter`와 `RoutingCache`를 각각
   `std::unique_ptr`로 주입받아 소유하며 (구체 구현체 선택은 호출자의
   몫), 이 둘에 대한 참조로 생성된 `Core`를 값으로 소유합니다.
   `Engine`의 세 override(`search`/`insert`/`remove`)는 `controller_`로 그대로
   위임(forward)할 뿐, 자체 로직이 없습니다.

2. **Core** -- `Core`, `AnchorManager`. `Core`는 index에 대해 알지 못하는
   제어 평면(control plane)으로, SEARCH/INSERT/DELETE가 *어디서*
   (Host/GPU/hybrid, 어떤 region에서) 실행될지를 결정합니다.
   `IAdapter`/`IRegion` 인터페이스와 `RoutingCache` 인터페이스에
   대해서만 작성되어 있으며, 구체적인 index나 구체적인 cache 구현에
   대해서는 절대 작성되지 않습니다. `Core`가 보유하는 것:
   - `IAdapter& adapter_` -- 주입받으며, 소유하지 않습니다.
   - `RoutingCache& routing_cache_` -- 주입받으며, 소유하지 않습니다.
     플러그 가능(pluggable)합니다: "이 query가 우리가 이미 본 것과
     가까운가"만 답합니다.
   - `AnchorManager anchor_manager_` -- 값으로 직접 소유하며, 플러그
     불가능합니다. Core 자신의 정책 상태로, 각 Anchor가 현재 보유한
     Stitch(write lease)들을 관리합니다.
   - `next_anchor_id_`, `drift_window_host_`, `drift_window_total_` --
     Core 자신의 작은 부기(bookkeeping) 상태들입니다 (id 할당, 추후
     drift 기반 승격 트리거를 위한 Host-vs-GPU traversal의 rolling
     카운터).

   `AnchorManager`는 작고 자기완결적인, mutex로 보호되는 map입니다.
   `RoutingCache`나 `IAdapter`, 그 이상의 어떤 것도 알지 못하며,
   오직 `VectorId -> Stitch` 부기만을 이해합니다.

3. **Adapter** -- `IAdapter`, `IRegion` (인터페이스만 존재하며, 현재
   이 트리에는 구체 구현체가 없습니다 -- 실제 index를 통합하는 작업은
   별도의 향후 작업으로 남겨두었습니다). `ASRoutingCacheHnsw`도 개념적으로
   이 레이어에 속합니다 (`RoutingCache` 인터페이스를 통해 `Core`에
   plug-in되는 구체 *구현체*입니다), vendored된 `thirdparty/hnswlib`
   서브모듈(릴리스 `v0.9.0`에 고정)을 기반으로 합니다. hnswlib은 `.cpp`
   전용 의존성입니다: `routing_cache_hnsw.hpp`는 중첩된 `Instance` 타입을
   forward-declare만 하고 `<hnswlib/hnswlib.h>`를 절대 include하지
   않으므로, 이 헤더만 include하는 코드는 hnswlib을 include path에 둘
   필요가 없습니다.

### 빌드 (`CMakeLists.txt`, `conda/arachne-blackwell.yml`)

- `project(arachne LANGUAGES CXX CUDA)`, C++20/CUDA20, 기본
  `CMAKE_CUDA_ARCHITECTURES`는 Blackwell(`100 120`)로 설정되어 있으며
  오버라이드 가능합니다. 의존성(`raft`, `spdlog`, `GTest`)은
  `CMAKE_PREFIX_PATH`를 통해 활성화된 conda 환경에서 해석됩니다.
- `hnswlib`은 `INTERFACE` 타겟입니다 (header-only, vendored, conda에는
  없음) `arachne_core`에 `PRIVATE`로 링크되므로, 소비자의 include path로
  절대 새어나가지 않습니다.
- `arachne_core`가 유일한 라이브러리이며, `interface/engine.cpp`,
  `core/core.cpp`, `core/anchor_manager.cpp`, `core/routing_cache_hnsw.cpp`로
  빌드됩니다. `arachne::core`로 alias됩니다.
- `ARACHNE_USE_RAFT` (기본 `ON`)는 로깅 백엔드를 전환하고 `raft::raft`를
  링크합니다; 꺼져있으면 standalone `spdlog`로 폴백합니다.
- `ARACHNE_BUILD_TESTS` (기본 `ON`)는 `test/`를 추가하며, GoogleTest
  (`GTest::gtest`, `GTest::gtest_main`, `gtest_discover_tests`로 discover)를
  기반으로 `arachne_tests`를 빌드합니다.
- `ASRoutingCacheHnsw`의 `std::thread`/`std::shared_mutex` 기반 백그라운드
  compaction을 위해 `Threads::Threads`가 public으로 링크됩니다.

### 포맷팅 규칙

`include/`와 `src/` 아래의 모든 수작성 파일(vendored된
`thirdparty/hnswlib` 서브모듈 제외)은 `CMakeLists.txt` 파일을 포함하여
**tab** 들여쓰기를 사용합니다.

## 3. 함수 수준 -- 핵심 구성요소들의 동작

### `Index` (`interface/index.hpp`)

```cpp
virtual SearchResult search(const Query& query) = 0;
virtual InsertResult insert(const Record& record) = 0;
virtual DeleteResult remove(VectorId id) = 0;
```

순수 인터페이스로, 상태도 `.cpp` 파일도 없습니다. virtual 소멸자와 세
개의 요청 메서드가 전부입니다. `RoutingCache`만큼이나 의도적으로
얇게(thin) 유지되어 있습니다: 호출자가 `Engine`이라는 특정 클래스가
아니라 "search/insert/remove를 할 수 있는 무언가"에 의존할 수 있도록
하기 위해서만 존재합니다.

### `Engine` (`interface/engine.hpp` / `.cpp`)

```cpp
Engine(std::unique_ptr<IAdapter> adapter, std::unique_ptr<RoutingCache> routing_cache);
SearchResult search(const Query& query) override;
InsertResult insert(const Record& record) override;
DeleteResult remove(VectorId id) override;
```

주입받은 두 의존성의 소유권을 가져오고, 이들에 대한 참조로 `controller_`를
생성합니다 (`controller_(*adapter_, *routing_cache_)`). 모든 public 메서드는
대응하는 `controller_` 메서드로 한 줄짜리 위임일 뿐, 자체 로직이 없습니다.

### `RoutingCache` (`core/routing_cache.hpp`)

정확히 하나의 질문에만 답하는 추상 인터페이스입니다 -- "들어온 query
vector가, 이전에 본 vector(`VectorId`로만 식별되는 Anchor) 중 GPU로
라우팅할만큼 충분히 가까운 것이 있는가?" Anchor id가 그 이상 무엇을
의미하는지는 전혀 알지 못합니다: Stitch/write-lease 부기도, eviction
정책도 여기에는 없습니다.

```cpp
virtual std::optional<VectorId> nearest(const VectorView& query) = 0;
virtual VectorId ensure(VectorId id, const VectorView& vector) = 0;
virtual void erase(VectorId id) = 0;
```

- `nearest` -- 가장 가까운 등록된 entry의 id, 혹은 충분히 가까운 것이
  없거나(또는 cache가 비어있으면) `nullopt`. "충분히 가깝다"의 판단은
  전적으로 구현체의 몫이며, `Core`는 자체 threshold를 적용하지 않습니다.
- `ensure` -- 멱등(idempotent)한 get-or-create입니다: 충분히 가까운
  entry가 이미 있으면 그 id를 반환하고, 없으면 `vector`를 `id`로 새로
  등록한 뒤 `id`를 반환합니다.
- `erase` -- entry를 제거합니다; `id`가 알려지지 않은 경우 아무 동작도
  하지 않습니다.

### `ASRoutingCacheHnsw` (`core/routing_cache_hnsw.hpp` / `.cpp`)

hnswlib을 기반으로 한 구체적인 `RoutingCache` 구현체로, raw index 위에
세 가지 책임이 얹혀 있습니다:

**동시성(Concurrency).** hnswlib 자체의 내부 locking은 동시 read끼리,
동시 write끼리는 서로 보호하지만, 동시 read와 동시 write 사이는
보호하지 *않습니다* (`searchKnn`은 hnswlib의 어떤 lock도 잡지 않은 채
link list를 순회합니다; `hnswalg.h` 소스를 직접 읽어 확인했습니다).
`ASRoutingCacheHnsw`는 자체 `std::shared_mutex mutex_`로 이를 보완합니다:
`nearest()`는 `shared_lock`을, `ensure()`/`erase()`는 `unique_lock`을
잡습니다. hnswlib은 이 두 lock 모드 중 하나 바깥에서는 절대 건드려지지
않습니다.

- `nearest(query)`: shared lock을 잡고 `active_->findNearest`로
  위임합니다.
- `ensure(id, vector)`: unique lock을 잡습니다; `findNearest`가 이미
  일치하는 것을 반환하면 그 id를 반환합니다 (요청받은 `id`는 버려집니다
  -- 호출자가 제안한 id는 기존에 충분히 가까운 entry가 없을 때만
  채택됩니다). 그렇지 않으면 `id`로 삽입한 뒤 `id`를 반환합니다.
- `erase(id)`: unique lock을 잡습니다; `active_->erase(id)`를 호출하며,
  이는 hnswlib에서 `markDelete`를 수행하고 tombstone 카운터를
  증가시킵니다. lock을 해제한 뒤, tombstone/(tombstone+live) 비율이
  `max_tombstone_ratio_`를 넘었으면 `triggerCompaction()`을 호출합니다.

**삭제 / Compaction.** hnswlib은 tombstone만 남길 뿐(`markDelete`) --
하부 라이브러리에서 실제 공간 회수는 일어나지 않습니다. tombstone
비율이 설정된 threshold를 넘으면, `ASRoutingCacheHnsw`는 백그라운드
스레드에서 active의 살아있는 entry들로부터 새로운 "shadow" `Instance`를
재구축한 뒤 교체합니다. `compactImpl()`은 세 단계로 동작합니다:

1. **Snapshot** -- 짧은 `shared_lock`; `active_->forEachLive`를 순회하여
   살아있는 모든 `(id, vector)` 쌍을 복사합니다. reader들과 다음
   writer는 이 복사 동안만 막히며, 아래의 재구축 동안은 막히지 않습니다.
2. **Rebuild** -- *lock을 전혀 잡지 않습니다*. `max(snapshot.size() * 2,
   initial_capacity_)` 용량으로 새 `Instance`를 생성하고, snapshot한
   모든 쌍을 여기에 삽입합니다. 이것이 비용이 큰 부분이며, 이 동안
   `active_`에 대한 read와 write 모두 완전히 동시에 진행됩니다.
3. **Reconcile and swap** -- 짧은 `unique_lock`. `active_`의 현재
   live-id 집합을 migration된 것과 비교합니다: rebuild 도중 `active_`에서
   삭제된 것은 shadow에서도 삭제하고, rebuild 도중 `active_`에 삽입된
   것은 shadow로 복사합니다. 그 뒤 `active_ = std::move(shadow)`합니다.
   이 단계의 비용은 index 크기가 아니라 rebuild 도중 변경된 양에
   비례합니다.

Stitch 부기가 `AnchorManager`로 옮겨갔기 때문에, compaction은 `(id,
vector)` 이상의 어떤 per-id 상태도 보존할 필요가 없습니다.

`triggerCompaction()`은 `compacting_.compare_exchange_strong(expected=false,
true)`를 사용하여 한 번에 최대 하나의 compaction 스레드만 실행되도록
합니다; 이미 실행 중일 때 중복으로 트리거되면 조용히 무시됩니다.
`waitForCompaction()`은 (`RoutingCache` 인터페이스에는 없는 메서드)
백그라운드 스레드를 join하며, 테스트와 정상적인 종료(graceful shutdown)
시점의 동기화 지점으로 사용됩니다. 소멸자에서도 스레드가 joinable하면
join합니다.

### `AnchorManager` (`core/anchor_manager.hpp` / `.cpp`)

`Core`가 넘겨주는 어떤 Anchor id에 대해서든 Stitch 부기를 담당합니다.
네 메서드 모두 단순한 `std::lock_guard<std::mutex>`를 사용합니다
(`ASRoutingCacheHnsw`와 달리 비용이 큰 백그라운드 연산이 없어 동시 reader를
통과시켜줄 shared/exclusive 구분이 필요 없습니다).

```cpp
std::vector<Stitch> stitchesOf(VectorId anchor_id) const;
void addStitch(VectorId anchor_id, RegionId region, LeaseHandle lease);
LeaseHandle removeStitch(VectorId anchor_id, RegionId region);
std::vector<Stitch> forget(VectorId anchor_id);
```

- `stitchesOf` -- 해당 anchor의 현재 Stitch 목록의 복사본을 반환합니다
  (없으면 빈 목록).
- `addStitch` -- region 단위로 멱등합니다: `anchor_id`가 이미 `region`에
  대한 Stitch를 갖고 있으면 아무 동작도 하지 않습니다 (기존 lease가
  유지되며 덮어쓰지 않습니다).
- `removeStitch` -- 한 region에 대한 Stitch를 제거하며, 호출자가 하부
  `IRegion` lease를 해제할 수 있도록 그 `LeaseHandle`을 반환합니다.
  제거할 것이 없으면 기본(invalid) handle을 반환합니다. anchor의 Stitch
  목록이 비게 되면 map entry 자체를 삭제합니다.
- `forget` -- 한 anchor의 *모든* Stitch를 한 번에 제거하고 반환합니다;
  향후 eviction 정책이 차가워진(cold) Anchor의 lease들을 `removeStitch`를
  반복 호출하지 않고 한 번에 회수할 때 사용할 지점입니다.

### `Core` (`core/core.hpp` / `.cpp`)

제어 평면입니다. 6개의 private 메서드가 Quick Summary의 4가지 설계
포인트에 직접 대응되며, `search`/`insert`/`remove`가 public 표면입니다.

```cpp
SearchResult search(const Query& query);
InsertResult insert(const Record& record);
DeleteResult remove(VectorId id);
```

- **`route(query) -> RoutingDecision`** (설계 포인트 1). `routing_cache_.nearest(query.vector)`를
  호출합니다; anchor id가 반환되면 `anchor_manager_.stitchesOf`로 그
  anchor의 Stitch를 조회합니다. 하나라도 있으면 query를 GPU-only 대상으로
  표시하고, 모든 Stitch의 region을 `predicted_scope`에 모읍니다.
- **`search(query)`**. `route()`를 호출합니다; 대상이면 `predicted_scope`로
  범위를 지정한 `GpuOnly` 모드의 `TraverseRequest`를 실행합니다. 이
  traversal이 `!completed_within_scope`를 보고하면, 호출자가 여전히
  완전한 답을 받을 수 있도록 두 번째 `Hybrid` 모드 traversal로
  폴백합니다. `recordTraversalForDrift`로 traversal 결과를 기록한 뒤,
  `routing_cache_.ensure(next_anchor_id_++, query.vector)`로 query
  vector를 (새로울 수 있는) Anchor로 등록합니다 -- 이는 검색이 끝난
  *이후*에 일어나므로, query는 자신이 도착하기 이전에 존재했던 Anchor를
  기준으로만 라우팅됩니다.
- **`insert(record)`**. 먼저 record의 vector에 대한 Anchor를
  보장합니다(`routing_cache_.ensure`). 그 anchor가 이미 유효한 Stitch를
  갖고 있으면, modification 요청을 그 단일 region으로 범위를 지정한
  `GpuOnly` 모드로 만들고 lease를 첨부합니다; 그렇지 않으면 요청은 scope
  없는 `Hybrid`로 남습니다. (발견된 첫 번째 유효한 Stitch만 사용합니다
  -- 다중 region insert는 명시적으로 향후 작업으로 남겨두었습니다.)
  `adapter_.modify()`가 성공하면, adapter가 실제로 수정했다고 보고한
  모든 region에 대해 `make(anchor_id, region)`을 호출합니다.
- **`remove(id)`**. `Delete` 모드의 `ModifyRequest`를 직접
  실행합니다; Anchor/routing과는 관련이 없습니다 (삭제는 근접도로
  라우팅할 vector가 아니라 id를 대상으로 합니다).
- **`recordTraversalForDrift(touched_host)`** (설계 포인트 2의 트리거).
  최근 N개의 traversal 중 몇 개가 Host를 건드렸는지를 rolling window
  (`kDriftWindowSize = 128`)로 유지하며, window가 차면 리셋합니다. 아직
  어떤 승격/축출 트리거도 이 값을 소비하지 않습니다 -- 카운터는
  존재하지만 아직 아무도 읽지 않습니다.
- **`promote(footprint)` / `evict(footprint)`** (설계 포인트 2, "언제"가
  아니라 "어떻게"). Placeholder 정책입니다: `promote`는 주어진
  footprint의 모든 region에 대해 `materializeOnDevice()`를 호출하고,
  `evict`는 `evictFromDevice()`를 호출합니다. 아직 어떤 트리거에도
  연결되어 있지 않습니다.
- **`verify(query, anchor_id, gpu_only_result)`** (설계 포인트 3). 아직
  `search()`에서 호출되지 않습니다. Ground truth로서 `Hybrid` 모드로
  query를 재실행하고 neighbor-id 시퀀스를 비교합니다. 불일치가 있으면
  `anchor_manager_.forget(anchor_id)`를 호출하여 그 anchor의 모든
  Stitch를 회수하고(해당 region들이 더 이상 anchor의 실제 locality를
  대표하지 않으므로), 회수된 각 lease를 `IRegion::releaseWriteLease`로
  해제합니다.
- **`make(anchor_id, region) -> bool`** (설계 포인트 4). 먼저
  `anchor_manager_.stitchesOf(anchor_id)`에서 `region`에 대한 기존
  Stitch가 있는지 확인합니다 -- 있으면 adapter를 건드리지 않고 즉시
  `true`를 반환합니다 (이 순서가 중요합니다: lease를 획득하기 *전에*
  확인함으로써, region이 이미 stitch되어 있는 경우 lease가 누수되는
  것을 방지합니다). 그렇지 않으면 region을 resolve하고, `Resident`
  상태를 요구하며, write lease를 획득하고, 성공하면
  `anchor_manager_.addStitch`로 Stitch를 기록합니다. region을 resolve할
  수 없거나, `Resident`가 아니거나, lease 획득이 실패하면 `false`를
  반환합니다.

**알려진, 문서화된 한계점**: `make()`의 check-then-acquire 순서는, 두
스레드가 동시에 같은 anchor를 같은 region에 stitch하려 할 때 TOCTOU
race window를 가집니다 -- 둘 다 "아직 stitch되지 않음" 체크를 통과한
뒤에야 `addStitch`를 호출할 수 있습니다. 결과는 무해한 중복
`acquireWriteLease()` 호출일 뿐 crash는 아닙니다; 실제 수정에는 필요한
per-(anchor, region) lock을 어디에 둘 것인지 결정이 필요하므로,
심각도가 낮다고 보고 아직 수정하지 않은 채로 남겨두었습니다.

### 로깅 (`logging.hpp`)

`ARACHNE_LOG_TRACE/DEBUG/INFO/WARN/ERROR(...)` 매크로이며, 어느 경우든
fmt 스타일의 `{}` placeholder를 사용합니다:

- `ARACHNE_WITH_RAFT`가 정의된 경우(RAFT가 링크된 경우): `RAFT_LOG_*`로
  위임되며, 이는 `raft::default_logger()` (rapids-logger/spdlog wrapper)를
  구동합니다 -- 그래서 Arachne의 control-plane 로그와 RAFT의
  GPU-primitive 로그가 하나의 sink/pattern/level을 공유합니다.
- 그렇지 않은 경우: standalone으로 지연 생성되는
  `arachne::default_logger()` (프로세스 전역 하나의
  `spdlog::stderr_color_mt("arachne")`)를 대상으로 `SPDLOG_LOGGER_*`로
  위임됩니다.

## 테스트

`test/routing_cache_hnsw_test.cpp` (10개 케이스)와
`test/anchor_manager_test.cpp` (9개 케이스), 둘 다 GoogleTest 기반이며,
`test/CMakeLists.txt`를 통해 하나의 `arachne_tests` 바이너리로 빌드됩니다.

특히: `ASRoutingCacheHnswTest.CompactionKeepsSurvivingIdsQueryableAndDropsErasedOnes`는
(낮은 `max_tombstone_ratio`를 통해) 실제 compaction을 유발시켜,
active/shadow swap 이후에도 살아남은 id는 여전히 질의 가능하고 삭제된
id는 되살아나지 않음을 검증합니다; `ConcurrentEnsureNearestEraseDoesNotRace`는
8개 스레드 x 200회의 혼합된 `ensure`/`nearest`/`erase` 연산을
실행하며, `ASRoutingCacheHnsw`의 `shared_mutex`가 방지하고자 하는 바로 그
read/write 위험을 잡아내기 위해 ThreadSanitizer 하에서 실행되는 것을
전제로 합니다.
