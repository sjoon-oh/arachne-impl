# HnswIndexDist 구현 현황 리포트

작성 시점 기준 코드 상태(`hnsw_index.hpp/.cpp`, `hnsw_index_dist.hpp/.cpp`,
`hnsw_dist_kernel.cu/.cuh`, `hnsw_index_anchor_entry.hpp/.cpp`)를 실제 소스와
1:1로 대조하여 정리. 코드 변경 없음 (조사/보고 전용).

## 0. 먼저: Layer-0 Traversal이 GPU에서만 도는가?

**아니다.** `HnswIndexDist::TraverseOneOnDevice()`(`hnsw_index_dist.cpp:50-218`)는
"distance-only offload"이지 "step-by-step 전체 GPU traversal"이 아니다.

- **Host에서 도는 부분**: 그래프를 실제로 걸어가는 제어 로직 전부.
  `candidate_set`(min-heap), `top_candidates`(max-heap), `visited` bitset,
  정지 조건(`current.first > top_candidates.top().first`), 다음 hop에 방문할
  이웃 목록 산출(`engineLevel0Neighbors()`) — 전부 host CPU의 `while` 루프
  (`hnsw_index_dist.cpp:169-199`) 안에서 순차 실행된다.
- **GPU로 넘어가는 부분**: 매 hop에서 새로 발견된 미방문 이웃들의 "거리 계산"
  만 하나의 배치로 묶여 `LaunchSquaredL2DistanceKernel()`로 넘어간다
  (`compute_distances` 람다, `hnsw_index_dist.cpp:100-145`).
- 매 hop마다 `cudaMemcpyAsync`(query 없으면 최초 1회, 후보 포인터 배열,
  결과) + `cudaStreamSynchronize()`가 발생 — hop 수만큼 host↔device 왕복이
  반복된다. 커널 하나 실행 후 바로 동기화하고 host로 돌아와 heap을 갱신한 뒤
  다음 hop을 결정하는 구조이므로, GPU는 "거리 계산기"로만 쓰이고 있고
  traversal의 제어권은 한 번도 GPU로 넘어가지 않는다.

즉, 이전 논의에서 나눴던 두 옵션 —
1) distance calculation만 GPU로,
2) traversal 자체(step-by-step)를 GPU로

— 중 **1번만 구현된 상태**이고, 2번(진짜 GPU-driven traversal, 즉 그래프
워크 자체를 커널 내부 루프로 구현하는 것)은 설계도 코드도 아직 없다.

---

## 1. hnswlib 코어 알고리즘과의 동일성

### 1.1 그대로 쓰는 부분 (hnswlib 소스 미수정, 알고리즘 재구현 없음)

`thirdparty/hnswlib`는 서브모듈 그대로이며 패치되지 않았다. `TypedHnswEngine`
(`hnsw_index.cpp:76-192`)은 hnswlib의 public API/멤버를 직접 호출하는 얇은
타입-소거 래퍼일 뿐이다.

| Arachne 쪽 진입점 | 실제 hnswlib 호출 | 위치 |
|---|---|---|
| `HnswIndex::traverseHost()` | `index_.searchKnnCloserFirst()` | `hnsw_index.cpp:126` |
| `HnswIndex::modifyHost()` Insert | `index_.addPoint()` | `hnsw_index.cpp:138` |
| `HnswIndex::modifyHost()` Delete | `index_.markDelete()` | `hnsw_index.cpp:151` |
| `HnswIndex::build()` | `index_.addPoint()` (루프) | `hnsw_index.cpp:180` |
| `HnswIndex::exportTo()` | `index_.saveIndex()` | `hnsw_index.cpp:159` |
| `HnswIndex::loadFrom()` | `index_.loadIndex()` | `hnsw_index.cpp:162` |
| 진입점 조회 | `index_.enterpoint_node_` | `hnsw_index.cpp:87` |
| level-0 이웃 조회 | `index_.get_linklist0()` / `getListCount()` | `hnsw_index.cpp:94-95` |
| 벡터 raw 포인터 | `index_.getDataByInternalId()` | `hnsw_index.cpp:90` |
| 삭제 여부 | `index_.isMarkedDeleted()` | `hnsw_index.cpp:103` |
| id 매핑 | `index_.label_lookup_` | `hnsw_index.cpp:110` |

→ `HnswIndex`, `HnswIndexAnchorEntry`의 host 경로는 **알고리즘 수준에서
hnswlib과 100% 동일**하다. 검색이면 검색, 삽입이면 삽입 모두 hnswlib 자체
함수 호출 결과를 그대로 감싸서 반환할 뿐, Arachne이 자체적으로 그래프
워크·힙 관리 등을 재구현한 부분이 host 경로에는 없다.

`HnswIndexAnchorEntry`는 현재 `resolveEntryPoint()`가 `TraverseRequest`에
`anchor_id` 필드가 없어 항상 `engineGlobalEntryPoint()`로 폴백한다
(`hnsw_index_anchor_entry.cpp:7-22`) — 즉 지금 이 클래스는 동작상
`HnswIndex`와 완전히 동일하며, anchor 캐시 로직(`anchor_entry_point_` 맵)은
채워지지도 조회되지도 않는 죽은 코드다. Core 쪽 변경(§10.4 point 1) 대기 중.

### 1.2 새로 재구현한 부분 — `HnswIndexDist`만 해당

`TraverseOneOnDevice()`는 hnswlib의 `searchBaseLayerST()`
(`thirdparty/hnswlib/hnswlib/hnswalg.h`)를 호출하지 않고 **처음부터
재구현**했다. 이유: hnswlib 검색 함수는 자기 `fstdistfunc_`를 인라인으로
호출하며, GPU 배치 거리 계산을 끼워넣을 seam이 없다(소스 패치 없이는).

구현된 알고리즘은 구조적으로 hnswlib의 level-0 greedy search와 동일한 형태
(min-heap candidate + bounded max-heap top-k + visited set)이지만, 다음
차이가 있다:

- **`ef` 파라미터가 없다** — `ef = top_k`로 고정 (`hnsw_index_dist.cpp:157`).
  hnswlib은 `ef_`를 top_k와 별도로 크게 잡아 recall을 올리는데, 이 구현은
  그 여유가 없어 동일 k에서 hnswlib 순정보다 recall이 낮을 수 있다. (정확성
  버그가 아니라 알려진 단순화 — 코드 주석에 명시됨.)
- 상위 레벨(level > 0) 디센트가 아예 없다 — `resolveEntryPoint()`가 반환하는
  진입점에서 바로 level-0 탐색을 시작한다. (`HnswIndex`의 host 경로도
  마찬가지로 상위 레벨은 hnswlib의 `searchKnnCloserFirst()` 내부에서
  처리되고 이쪽은 재노출되지 않으므로, `HnswIndexDist`가 상위 레벨을
  건너뛰는 것 자체는 "덜 구현됨"이 아니라 애초에 GPU 경로가 level-0만
  대상으로 설계됨.)
- Float32/L2 조합만 지원 (`hnsw_index_dist.cpp:51-55`에서 그 외는
  `std::logic_error`).

파일 상단 주석(`hnsw_index_dist.hpp:11-23`)에 "왜 hnswlib을 그대로 못
쓰는지"가 명시되어 있어, 사용자가 요청한 "변경 부분을 코드로 남겨두기"
조건은 충족된 상태.

### 1.3 CUDA 커널 (`hnsw_dist_kernel.cu`)

hnswlib에 대응하는 게 없는 완전히 새로운 코드 — hnswlib은 CPU distance
function(`fstdistfunc_`, SIMD)만 갖고 있고 GPU 커널이 없으므로 "동일하게
유지"할 대상 자체가 없다. Squared L2, candidate당 CUDA block 1개, 128
threads/block, shared-memory tree reduction. 첫 커널이라 정확도·형태는
검증됐지만(§3 참고), 성능 최적화(예: 여러 후보를 하나의 block에 묶기,
half-precision 등)는 전혀 하지 않은 가장 단순한 형태.

---

## 2. Region 등록/사용 범위

### 2.1 Region 단위

`HnswIndex::BuildRegions()`(`hnsw_index.cpp:273-283`)가 hnswlib의
`data_level0_memory_` — level-0 레코드(그래프 링크 + 벡터 데이터 + 라벨이
한 덩어리로 붙은 고정 크기 레코드)들이 연속으로 배치된 블록 — 을
`vectors_per_region`개씩 id-range로 슬라이스한다. 슬라이스 하나 = `HnswRegion`
하나.

```cpp
for (std::size_t start = 0; start < capacity_; start += vectors_per_region_) {
  std::size_t count = std::min(vectors_per_region_, capacity_ - start);
  void* ptr = base + start * record_bytes;
  regions_.push_back(std::make_unique<HnswRegion>(region_id, ptr, count * record_bytes, record_bytes));
}
```

`subregion_bytes`가 `record_bytes`(레코드 하나 크기)로 설정되어 있어
(`hnsw_index.cpp:281`), Controller의 dirty-bitmap 추적 메커니즘
(`gpu/dirty_header.hpp`)이 이 Region에 대해 활성화된다 — device 버퍼 앞에
dirty-header가 붙는다는 뜻이고, 이게 실제로 GPU 버그 #1(device pointer
오프셋 계산 누락)의 원인이었다.

### 2.2 등록 범위: Level-0만, 상위 레벨은 Region이 아님

`registerAllRegions()`(`hnsw_index.cpp:374-377`)가 Controller에 등록하는
건 **level-0 레코드 전체뿐**이다. hnswlib 내부에는 level-0과 별도로 상위
레벨 링크를 담는 `linkLists_` 배열이 존재하는데, 이건 애초에
`HnswRegion`/Region 개념으로 노출되지 않는다 — Region으로 만들 코드 자체가
없다.

결과적으로:
- 지금 GPU offload가 다루는 범위는 **level-0 그래프뿐**이다.
- 예전에 논의했던 "상위 레벨은 상시 GPU-resident, level-0만 dynamic"
  아이디어는 **아직 구현되지 않았다** — 상위 레벨이 Region으로 존재하지
  않으므로 애초에 GPU에 올릴 대상 자체가 없는 상태다. 이건 이 리포트가
  다루는 코드 범위에서 완전히 미착수 항목.
- `HnswIndexDist`가 `resolveEntryPoint()`로 얻는 진입점부터 바로 level-0을
  타는 것도 이와 일치 — 상위 레벨을 "스킵"하는 게 아니라 애초에 상위
  레벨을 다룰 GPU 경로 자체가 없다.

### 2.3 Residency 체크의 정직성

`HnswIndexDist::TraverseOneOnDevice()`는 후보 배치마다
`Controller::acquireRegion()`으로 실제 residency를 확인하고
(`hnsw_index_dist.cpp:107-108`), 하나라도 GPU에 없으면 그 자리에서
`completed_within_scope=false`로 멈춘다 — host 메모리를 몰래 읽어서 계속
진행하지 않는다. 이 부분은 설계 문서(§9.3)와 정확히 일치하게 구현되어
있다.

### 2.4 알려진 미해결 이슈

`hnsw_index.hpp:45-49` 주석에 명시: hnswlib의 internal id는 삽입 순서로
부여되지 공간적으로 인접하게 부여되지 않는다. 따라서 id-contiguous Region이
그래프상으로도 인접한 노드들을 묶는다는 보장이 없다 — Region locality
품질(즉 "이 Region을 GPU에 올리면 실제로 몇 hop이나 그 안에서 끝나는가")은
전혀 측정되지 않았다.

---

## 3. 검증 수준

### 3.1 Host 경로 (`hnsw_index_test.cpp`)

`HnswIndex`/`HnswIndexAnchorEntry`, 4개 dtype(Int8/UInt8/Float16/Float32) ×
dim=16, gtest 파라미터화, 각 케이스 최대 200벡터 규모:

| 테스트 | 확인 내용 |
|---|---|
| `BuildThenSearchSelfRecall` | 자기 벡터로 검색 시 자기 자신이 나오는지, ≤10/200 miss 허용 (hnswlib 자체 dtype 테스트와 동일 기준) |
| `InsertThenSearchFindsNewVector` | Insert 직후 검색에 새 벡터가 잡히는지 |
| `DeleteThenSearchOmitsDeletedVector` | Delete 후 검색 결과에서 제외되는지 |
| `ExportLoadRoundTripPreservesSearchResults` | Export→Load 후 20개 쿼리의 top-3 결과가 동일한지 |
| `LoadRejectsCapacityMismatch` | capacity 불일치 파일 로드 시 예외 발생 확인 |
| `MatchesHnswIndexSearchResultsBeforeAnchorIdLands` | `HnswIndexAnchorEntry` == `HnswIndex` 검색 결과 동일성 (anchor 로직은 죽은 코드이므로 사실상 회귀 방지용) |

### 3.2 GPU 경로 (`hnsw_index_dist_test.cpp`)

단일 테스트 `TraverseDeviceMatchesTraverseHostOnResidentRegions`:
- Float32/L2, **dim=16, 200벡터, capacity 300** — 소규모
- 실제 `Controller` + `ASRoutingCacheHnsw`로 `controller.search()`를 200회
  돌려 자연스러운 promotion을 유도, GPU 예산은 전체 host 바이트의 2배(모든
  Region이 상주 가능하도록 넉넉하게 설정)
- 5개 간격으로 20개 쿼리에 대해 `traverseHost()`(hnswlib 순정) vs
  `traverseDevice()`(GPU offload) 결과를 id + 거리(1e-2 허용오차) 비교
- `completed_within_scope=false`인 케이스는 skip하되, 최소 1개는 실제로
  GPU 완료됐는지 assert
- 이 테스트로 실제 버그 2개(dirty-header 오프셋 누락, CUDA 아키텍처
  SM75 미포함으로 인한 커널 무동작)를 찾아 수정 완료, 현재는 통과.

### 3.3 검증되지 않은 것 (사용자 요청 4번 항목, 완전 미착수)

- **규모**: 128 dimension, 10M 스케일 — 현재 테스트는 dim=16, 최대
  200~300벡터. 오더가 다섯 자리 이상 차이난다.
- **Multi-thread**: 현재 GPU 테스트는 단일 스레드 순차 호출뿐. 동시 요청 시
  `mutex_` 경합(호스트 루프 전체 + GPU 커널 launch/sync를 통째로 잠금 —
  `hnsw_index_dist.cpp:68`)이 실질적 병목이 될 가능성이 있는데, 이건
  측정된 적이 없다.
- **Float32 normalize**: 요청된 "float32는 normalized"가 아직 테스트
  데이터 생성기(`GenerateVectors()`)에 반영 안 됨 — 정규화 로직 자체가
  없음.
- **`build()`의 확장성**: 현재 `build()`는 hnswlib `addPoint()`를
  단일스레드로 순차 호출(`hnsw_index.cpp:173-182`) — "TEMPORARY"로 명시된
  placeholder. 10M 스케일에서 이 자체가 빌드 시간의 지배적 병목이 될 수
  있으나 아직 측정되지 않음.

### 3.4 요약 (이 시점 기준, §4에서 갱신됨)

Host 경로는 **기능·알고리즘 모두 hnswlib과 동일함이 다중 dtype 소규모
테스트로 확실히 검증**됐다. GPU 경로는 **정확성만 단일 소규모 케이스로
확인**됐고, 성능·확장성(멀티스레드, 대규모)은 전혀 측정되지 않았다. Region은
level-0에만 걸려 있고 상위 레벨은 아직 Region 개념 자체가 없다. 사용자가
원래 요청한 "10M 스케일 + normalize + multi-thread excessive test"는 코드가
전혀 없는 상태 — 다음 작업 대상.

---

## 4. 후속 구현: Anchor 기반 진입점 + Beam-width(a) — hnsw_index_dist / hnsw_index_anchor_entry 재구성

§0~3의 리뷰에서 나온 두 가지 논의(고정 entry point 문제, single-round-trip
vs beam-width 확장 논의)를 바탕으로 다음을 구현했다. Bounded BFS(§리뷰의
"(b)" 옵션, entry point 주변 N-hop을 통째로 긁어 한 번에 계산)는 **다른
알고리즘으로 판단해 구현하지 않았다** — 사용자 지시대로 beam-width 확장
("(a)")만 구현.

### 4.1 클래스 구조 변경

기존에는 `HnswIndex` 아래 `HnswIndexDist`(GPU distance-offload)와
`HnswIndexAnchorEntry`(entry point만 오버라이드하는 스켈레톤, 실제로는
`resolveEntryPoint()`가 아무 데서도 호출되지 않는 죽은 코드였음)가 **각각
독립적인 형제 클래스**였다. 이제:

```
HnswIndex (concrete, host-only, hnswlib as-is)
  └─ HnswIndexDist (naive GPU offload: 고정 global entry, beam width=1)
       └─ HnswIndexAnchorEntry (anchor 기반 entry point 캐시 + beam width>1)
```

`HnswIndexAnchorEntry`가 `HnswIndexDist`를 **상속**하도록 바꿨다 (기존
`HnswIndex` 직속이 아님). 이유: 두 클래스가 필요로 하는 GPU 오프로드 배관
(`TraverseOneOnDevice()`의 CUDA 호출, `Controller::acquireRegion()` residency
체크, dirty-header 보정 주소 계산)이 완전히 동일하고, 다른 건 오직
"entry point를 어떻게 고르는가"(`resolveEntryPoint()`, 이미 virtual)와
"한 라운드에 후보를 몇 개 pop하는가"(`BeamWidth()`, 새로 추가한 virtual
hook, 기본값 1)뿐이기 때문이다. `HnswIndexAnchorEntry`는 이 두 hook만
오버라이드하고, `traverseDevice()`/`TraverseOneOnDevice()` 자체는 전혀
재정의하지 않는다 — 상속을 통해 그대로 재사용.

`HnswIndexDist`는 이제 "naive 버전"(항상 global entry point, beam width=1
— 정확히 이전과 동일한 동작)으로 자리매김했고, `HnswIndexAnchorEntry`가 그
위에 두 가지를 얹은 "개선 버전"이다.

### 4.2 Beam-width(a) 구현

`HnswIndexDist::BeamWidth() -> size_t`(기본 1)를 protected virtual로 추가.
`TraverseOneOnDevice()`의 루프를 "매 라운드 정확히 1개 pop"에서 "매 라운드
최대 `BeamWidth()`개 pop, 그 이웃들을 합쳐 GPU 호출 1번"으로 일반화
(`hnsw_index_dist.cpp`). Pop 대상 선정 기준(정지 조건)은 바뀌지 않았다 —
"이번 라운드에 몇 개를 한꺼번에 배치로 묶느냐"만 다르고, 어떤 후보가
확장 가능한지의 판정 자체는 B=1일 때와 동일한 규칙을 그대로 씀. `B=1`일
때 기존 동작과 완전히 동일함을 `hnsw_index_dist_test.cpp`(변경 없이 그대로
통과)로 확인.

`HnswIndexAnchorEntry::BeamWidth()`는 생성자 인자로 받은 `beam_width_`
(기본값 `kDefaultBeamWidth=4`, 튜닝된 값은 아니고 합리적인 시작값)를 반환.

### 4.3 Anchor 기반 진입점 캐시

**설계상 중요한 정정**: 처음에는 "Anchor id를 hnswlib의 `label_lookup_`에
직접 조회하면 되지 않을까"로 접근했으나, `Controller`/`RegionManager` 코드를
직접 읽어 확인한 결과 **틀린 가정**이었다 — 검색(`Controller::search()`)이
триggер하는 Hybrid 승격의 Anchor id는 `next_anchor_id_`라는 **순수 카운터**
이고, RoutingCache에 등록되는 대표 벡터는 **쿼리 벡터 자신**이지 실제
저장된 데이터셋 벡터가 아니다(`controller.cpp`의 `search()`/`dispatch()`
참고). 즉 Anchor id는 메인 그래프의 실제 internal id로 일반적으로 변환되지
않는다 — 사용자가 원래 표현한 "anchor와 진입점(진짜 저장된 벡터)은 다를 수
있다"는 지적이 정확히 이 지점이었다.

그래서 최종 구현은 원래 스켈레톤이 의도했던 방식(캐시 맵)으로 돌아갔다:

- `HnswIndexAnchorEntry::traverseHost()`를 오버라이드해서, `request.anchor_id`가
  실려 있고 검색 결과가 있으면, **그 검색이 실제로 찾아낸 top-1 결과의
  internal id**를 `anchor_entry_point_[anchor_id]`에 기록한다 (검색
  자체는 `HnswIndex::traverseHost()` 그대로 위임 — hnswlib 결과 자체는
  전혀 안 바뀜).
- `resolveEntryPoint()`는 `request.anchor_id`가 캐시에 있으면 그 internal
  id를, 없으면 기존처럼 `engineGlobalEntryPoint()`를 반환.
- 새 protected 헬퍼 `HnswIndex::engineInternalIdFor(VectorId) -> optional<uint32_t>`
  를 추가(`hnsw_index.hpp/.cpp`) — top-1 결과의 external id를 internal id로
  역변환하는 데만 쓰임(hnswlib이 이미 갖고 있는 `label_lookup_` 그대로
  사용, 재구현 없음).

**Core 변경 (사용자 승인 범위 내, `[코드 수정 O]`)**:
- `TraverseRequest`(`adapter/index_adapter.hpp`)에 `std::optional<VectorId> anchor_id`
  필드 추가. Core는 이 값을 전혀 해석하지 않고 adapter까지 그대로 전달만
  한다(`OpaqueData`와 같은 통과 전용 원칙).
- `Controller::RoutingDecision`에 `anchor_id` 추가, `route()`가
  `routing_cache_.nearest()` 히트 시 채움.
- `routeSearch()`/`insert()`가 이 값을 `TraverseRequest::anchor_id`로 전달.
- **버그 발견 및 수정**: 최초 구현 후 실제 `Controller`로 end-to-end
  테스트해보니 캐시가 **한 번도 채워지지 않는** 구조적 문제를 발견했다 —
  RoutingCache에 해당 locality가 처음 등록되는 바로 그 Hybrid 호출은
  `decision.anchor_id`가 아직 비어있어(`route()`가 처음 보는 쿼리라서)
  `traverseHost()`가 `anchor_id=nullopt`로 호출되고, 캐시를 채울 기회가
  없었다. `Controller::search()`/`insert()`에 작은 수정을 추가해서, 이번에
  새로 발급하는 Anchor id(= `requestPromotion()`이 등록에 쓸 그 id)를
  `TraverseRequest::anchor_id`에도 즉시 실어 보내도록 했다 — 이렇게 하면
  이번 Hybrid 완료가 캐시를 채운 바로 그 id가, 나중에 같은 locality를
  다시 조회했을 때 `nearest()`가 돌려주는 id와 정확히 일치해서 캐시
  히트가 실제로 발생한다.

### 4.4 테스트

- 기존 `hnsw_index_dist_test.cpp`(naive, B=1)는 변경 없이 그대로 통과 —
  일반화가 기존 동작을 정확히 보존함을 확인.
- `hnsw_index_test.cpp`의 `HnswIndexAnchorEntryTest`를 새 의미에 맞게
  갱신(`HostSearchMatchesPlainHnswIndexRegardlessOfAnchorId`) — host 경로는
  anchor_id 유무와 무관하게 여전히 `HnswIndex`와 동일한 결과를 내는지 확인
  (host 경로는 `resolveEntryPoint()`를 애초에 안 쓰므로).
- 신규 `hnsw_index_anchor_entry_dist_test.cpp`
  (`HnswIndexAnchorEntryDistTest.TraverseDeviceUsesCachedAnchorEntryPointAndStaysApproximatelyAccurate`):
  Float32/L2, dim=16, **2000벡터** 스케일(기존 GPU 테스트의 200벡터보다
  10배), 실제 `Controller` 사용.
  - **1차 패스**: 67개 쿼리(30개 간격 self-query)를 실제 `controller.search()`로
    실행 — 처음 보는 locality라 Hybrid로 처리되며 캐시가 채워짐. (테스트
    설계상 발견한 이슈: 67개를 한꺼번에 쏘면 같은 Region을 다투는 Anchor가
    많아져 대부분 `Deferred`로 밀려나고 재시도 없이 드롭됨 — 쿼리마다
    `waitIdle()`로 정착시켜 우회.)
  - **2차 패스**: 동일 쿼리 재실행 — `served_gpu_only` 쿼리가 실제로
    발생하는지 확인(`ASSERT_GT(served_gpu_only, 0u)`), 그리고
    `traverseHost()`(hnswlib 순정 ground truth) 대비 top-5 overlap 비율이
    무작위 수준(거의 0)보다 뚜렷이 높은지 확인(`overlap_ratio >= 0.3`) —
    "정확히 일치하진 않아도 비슷한 무언가"라는 기준을 정량화.
  - **결정론적 단위 확인**: Controller의 승격 타이밍과 무관하게, 한 번도
    쓰인 적 없는 `anchor_id`로 `traverseHost()`를 호출해 캐시를 심고, 곧바로
    같은 `anchor_id`로 `traverseDevice()`를 호출해서 (완료된 경우) 두
    결과가 겹치는지(`Overlap(...) > 0`) 확인 — 캐시 메커니즘 자체를
    Controller의 확률적 승격 타이밍과 분리해서 검증.
  - 전 항목 통과, 3회 재실행에서도 안정적(고정 RNG seed).
- 전체 스위트 324개 중 323개 통과 — 유일한 실패는 이 세션 이전부터 있던,
  이 작업과 무관한 `ControllerGpuResidencyTest.PromoteEvictsMultipleVictimsWhenOneIsNotEnoughCapacity`.

### 4.5 아직 남은 한계

- `anchor_entry_point_` 맵은 **무한정 커지고 절대 축출되지 않는다** — Anchor가
  Controller 쪽에서 evict돼도 이 캐시에는 알림이 안 온다. 실험적 첫 구현
  범위에서는 허용, 프로덕션화 시 해결 필요.
- Beam width 기본값(4)은 튜닝되지 않은 값 — 실제로 왕복 횟수 대비 정확도
  트레이드오프가 어떤지 측정되지 않았다.
- 여전히 §0~3에서 지적한 두 가지 근본 한계(상위 레벨 디센트 없음, `ef=top_k`
  고정)는 그대로 남아있다 — 이번 작업은 "고정 entry point" 문제를 anchor
  캐시로 완화했을 뿐, 상위 레벨 디센트 자체를 구현한 것은 아니다.
- 10M 스케일 · float32 정규화 · multi-thread excessive test는 여전히
  미착수.
