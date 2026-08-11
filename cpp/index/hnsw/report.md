# HNSW GPU 포팅 고려사항 (draft)

> 작성: 2026-08-07
> 범위: `cpp/thirdparty/hnswlib`를 참고/기반으로, Arachne의 `IAdapter`/`IRegion`을 구현하는
> 새 HNSW 어댑터(`cpp/index/hnsw` 아래에 C++로 구현 예정)를 GPU Region 모델 위에 올리기 위한
> 사전 조사. 코드 구현은 포함하지 않는다 — 결정이 필요한 지점과 대안을 나열하는 것이 이 문서의
> 목적이다.
>
> **범위 밖 명시**: 이 문서는 현재 `src/core/as_routing_cache_hnsw.cpp`가 hnswlib을 감싸서
> 쓰고 있는 `ASRoutingCacheHnsw`(RoutingCache의 Active/Shadow 구현체)와는 다른 대상을 다룬다.
> 그건 Anchor 식별용 소규모 인덱스이고, 여기서 다루는 건 전체 데이터셋을 담당하는 실제
> `IAdapter` 구현이다 — `doc/code-documentation-v3.md` §28이 "현재 구현 경계"로 명시한
> "실제 production GPU-native ANN adapter와 kernel" 부재를 메우는 후보 중 하나.

---

## 0. 왜 이 문서가 필요한가

Arachne의 Region 모델(`include/adapter/region.hpp`)은 "adapter가 자기 상태를 Region 단위로
나눌 수 있어야 한다"는 것을 전제로 한다. hnswlib은 GPU를 염두에 두지 않고 설계된 라이브러리이므로,
포팅은 두 가지를 각각 따로 물어야 한다.

1. **Region 단위 관리** — hnswlib의 메모리 레이아웃이 "연속된 바이트 범위 = 하나의 Region"이라는
   Arachne의 가정과 얼마나 잘 맞는가.
2. **GPU 연산 수행** — hnswlib이 실제로 하는 계산(그래프 탐색, 거리 계산, 이웃 재배선)을 GPU
   커널로 옮기는 게 알고리즘적으로 가능한가.

두 질문의 답은 서로 다르다 — 아래에서 구분해서 다룬다.

---

## 1. 현재 hnswlib 구조 요약 (근거 코드)

| 구성요소 | 위치 | 레이아웃 |
| --- | --- | --- |
| `data_level0_memory_` | `hnswalg.h:50`, 할당 `hnswalg.h:126` | `max_elements_ * size_data_per_element_` bytes, **하나의 연속 `malloc` 블록**. 노드 하나의 레코드 = `[level0 링크리스트][벡터 데이터][label]` (`hnswalg.h:120-124`: `offsetLevel0_=0`, `offsetData_=size_links_level0_`, `label_offset_=size_links_level0_+data_size_`) |
| `linkLists_` (level > 0) | `hnswalg.h:51` | `char**` — 원소마다 **별도로 malloc된** 블록 (`hnswalg.h:1206`, level 0가 아닌 원소만). 연속 배열이 아님 |
| `link_list_locks_` | `hnswalg.h:43` | 원소당 1개 `std::mutex`. Insert 시 새 노드뿐 아니라 **선택된 이웃 노드들의 락도 함께 잡고 그 이웃의 링크리스트를 수정**한다 (`mutuallyConnectNewElement`, `hnswalg.h:526-627`, 특히 554-627) |
| `global` | `hnswalg.h:42` | entry point/max level 갱신을 직렬화하는 단일 mutex (`hnswalg.h:1192`) |
| `visited_list_pool_` | `hnswalg.h:37`, `visited_list_pool.h` | 쿼리 1회당 `max_elements_` 크기의 재사용 가능한 방문-비트셋. 전체 인덱스 크기에 비례, Region과 무관하게 global |
| 탐색 (`searchBaseLayer*`) | `hnswalg.h:225-440` | Greedy best-first: 후보 큐에서 최근접 미방문 노드를 꺼내 그 이웃들과 거리 계산 → 큐에 삽입, 반복. **다음에 방문할 노드가 현재 스텝 결과에 의존**하는 순차적 루프 |

이미 현재 저장소 안에 이 구조를 실제로 사용하는 코드가 있다 —
`src/core/as_routing_cache_hnsw.cpp:36-38`의 `TypedInstance` 생성자가
`hnswlib::HierarchicalNSW<DistT>(&space_, capacity, M, ef_construction, ...)`를 그대로
인스턴스화한다. 다만 이건 Anchor 식별용(소규모)이고 Region 관리 대상이 아니다.

---

## 2. 가능한 부분

### 2.1 Region 단위 관리 관점

- **`data_level0_memory_`는 이미 하나의 연속 배열**이라, internal id의 연속 구간을 Region
  경계로 삼는 것 자체는 자연스럽다. 게다가 `region.hpp:48-55`의 `HostRegionView::subregion_bytes`
  주석은 이미 *"roughly one graph node's record for an HNSW-style adapter"*라고 명시하고
  있다 — 즉 `stride = size_data_per_element_`로 놓는 선택은 기존 설계가 이미 예견한 것이고,
  `gpu/dirty_header.hpp`의 dirty-bitmap 메커니즘을 그대로 재사용할 수 있다는 뜻이다.
- Level-0 링크리스트가 노드 레코드 안에 포함돼 있으므로, **"level 0 그래프만 우선 Region/GPU
  대상으로 삼고 상위 레벨은 host 전용으로 남긴다"**는 절충이 가능하다 — 탐색 품질의 대부분은
  level 0 그래프가 담당하고, 상위 레벨은 진입점(entry point)을 대략적으로 좁히는 역할이라
  상대적으로 접근 빈도가 낮다.

### 2.2 GPU 연산 수행 관점

- 탐색 루프에서 **"현재 프론티어 노드의 이웃들과 거리 계산"** 스텝은 이웃 수(M 또는 M0)만큼
  서로 독립적이다 — 이건 warp/block 단위 병렬화의 자연스러운 단위다 (쿼리 1개 = warp/block
  1개, 이웃 각각을 스레드에 매핑). 이 패턴은 SONG/GGNN 같은 GPU 그래프 ANN 연구와
  NVIDIA RAFT의 CAGRA가 실제로 쓰는 방식과 같다.
- 흥미롭게도 **이 저장소 자체가 이미 이 방향을 언급하고 있다**: `include/adapter/index_adapter.hpp:103-104`의
  `traverseDevice()` 설계 주석이 "e.g. one kernel launch, **RAFT/CAGRA-style**"라고 명시한다.
  즉 GPU-native traverse를 CAGRA 방식으로 짜는 것은 원 설계가 이미 상정한 그림이다.
- **Search와 Insert를 다른 실행 전략으로 분리하는 것**은 Arachne 프레임워크 차원에서 이미
  1급 시민이다: `IAdapter::modifyDevice()`의 기본 구현은 `std::logic_error`를 던진다
  (`index_adapter.hpp:160-164`) — "GPU search는 구현하되 GPU insert는 구현하지 않는 adapter"가
  프레임워크가 이미 정상 케이스로 취급하는 형태다. 아래 §3에서 보듯 HNSW의 insert는 GPU
  동시성 관점에서 search보다 훨씬 어렵기 때문에, 이 분리는 실질적인 이점이 있다.
- 참고로 이미 존재하는 `test/stress/stress_index.hpp`(brute-force test double)의
  `traverseDevice()`는 **실제 GPU 커널이 아니라 host 버퍼를 그대로 재사용하는 stand-in**이라는
  주석이 달려 있다(`stress_index.hpp:101-105`, `"until stage 4 adds a real write kernel"`).
  즉 **현재 이 저장소에는 실제로 device에서 도는 ANN 연산 커널이 하나도 없다** — HNSW 포팅
  이전에, "GPU 커널이 Arachne의 Region/lease/generation 모델과 맞물려 정상 동작하는가"
  자체가 아직 어디서도 검증되지 않은 상태라는 점은 감안해야 한다.

---

## 3. 반드시 결정이 필요한 부분과 대안

### 결정 1 — hnswlib을 그대로 쓸 것인가, 새로 작성할 것인가

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. 그대로 wrap | `HierarchicalNSW`를 `traverseHost`/`modifyHost` baseline으로 그대로 사용, GPU는 완전히 별도 자료구조로 새로 작성 후 두 자료구조를 동기화 | 작업량 적음. 그러나 host/device 두 벌의 그래프를 별도로 유지·동기화해야 하는 복잡도가 생김 |
| B. hnswlib을 fork/patch | `linkLists_`까지 연속 arena로 재구성(2.1의 아이디어 확장), 알고리즘 로직은 그대로 유지 | 알고리즘 검증 부담이 적음(로직 재사용). 다만 thirdparty를 fork하면 upstream과 diverge, 유지보수 비용 발생 |
| C. Region-native로 새로 작성 | hnswlib은 참고자료로만 쓰고 `cpp/index/hnsw`에 처음부터 설계 | 가장 깨끗하게 Arachne 모델과 결합. 작업량/검증 부담 최대 |
| D. HNSW 포팅 대신 GPU-native 인덱스로 대체 | RAFT/cuVS의 CAGRA(그래프 기반, GPU-native)를 `IAdapter`로 감싼다 — "포팅"이 아니라 "교체" | 이미 `raft::raft`가 하드 의존성(`CMakeLists.txt`)이라 도입 장벽이 낮음. 다만 HNSW 자체의 특성(증분 삽입 친화적, 삭제 처리 등)을 잃을 수 있음 — CAGRA는 배치/오프라인 그래프 구축에 가깝다 |

### 결정 2 — 상위 레벨(level > 0) 링크리스트를 어떻게 Region화할 것인가

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. 레벨0 레코드에 고정 슬롯으로 흡수 | 모든 노드에 대해 최대 레벨만큼의 링크 공간을 미리 예약 | 레이아웃 단순, Region 경계 계산 쉬움. 그러나 `mult_ = 1/ln(M)` 확률로 대부분의 노드가 level 0에서 멈추므로(`hnswalg.h:142`) 공간 낭비가 큼 |
| B. 상위 레벨 전용 별도 Region 풀 | level별 혹은 전체 상위 레벨용 연속 arena + 노드→오프셋 간접 테이블 | 공간 효율적. 간접 테이블 자체의 관리/동기화 비용 추가 |
| C. 상위 레벨은 host 전용 유지 | GPU/Region 대상에서 제외, 진입점 근사 용도로만 host에서 처리 | 구현 가장 단순, §2.1의 "level 0만 우선" 절충과 자연스럽게 맞음. 상위 레벨 접근도 어차피 top-down 진입 단계에서 소수 hop뿐이라 host 왕복 비용이 상대적으로 작을 가능성 |

### 결정 3 — cross-region write(이웃 재배선)를 어떻게 반영할 것인가

hnswlib의 `mutuallyConnectNewElement`(`hnswalg.h:506-630`)는 삽입 시 새 노드뿐 아니라
**선택된 이웃들의 링크리스트도 함께 수정**한다. Region을 internal-id 연속 구간으로 나누면,
그래프상 이웃이 id상 이웃이라는 보장이 없으므로 **삽입 1건이 여러 Region에 걸쳐 쓰기를
발생시킬 수 있다**. 이는 이미 `region.hpp:62-65`의 `ReconciliationReport{closed,
touched_neighbors}`와 `index_adapter.hpp:75`의 `ModifyResult::modified` footprint가
정확히 이런 상황을 염두에 두고 설계돼 있다는 점은 참고할 만하다.

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. 기존 `ReconciliationReport`/`touched_neighbors` 그대로 사용 | 영향받은 모든 이웃 Region을 `modified` footprint에 포함, 각 Region이 자기 lease/dirty를 개별 관리 | 기존 계약과 정합적. 삽입 1건이 다건의 Region write-lease를 동시에 필요로 하게 됨 — lease 획득 순서/실패 처리를 정의해야 함 |
| B. 지연 reconcile | insert는 "제안"만 하고, 실제 이웃 리스트 갱신은 별도 background 패스에서 일괄 적용 | GPU/batch 친화적(hnswlib의 즉시 갱신과 다른 모델). 갱신 반영 전까지 그래프가 일시적으로 "덜 정확"한 구간이 생김 |
| C. 이웃 후보를 같은 Region 내부로 제한 | 삽입 시 이웃 탐색 범위를 자기 Region으로 한정 | 구현 단순, cross-region write 자체가 사라짐. recall/그래프 품질 저하 리스크 — 검증 필요 |

### 결정 4 — 동시성 모델(`link_list_locks_`)을 GPU에 어떻게 이식할 것인가

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. Batch/offline 재구축 | fine-grained lock을 버리고 삽입을 모아 주기적으로 그래프 일부를 재계산 | Arachne의 Coordinator/relocation-batch 패턴과 결이 비슷해 통합이 자연스러움. 삽입 반영 지연 발생 |
| B. GPU atomic CAS 기반 lock-free 갱신 | 버전 스탬프 + CAS로 이웃 리스트 갱신 | hnswlib의 즉시-갱신 모델을 가장 가깝게 재현. 경쟁이 심하면 성능 저하 크고 구현/검증 난이도 높음 |
| C. Insert는 Host 전용 고정 | GPU는 Search만 담당(§2.2 참고), Insert는 항상 Hybrid | 프레임워크가 기본으로 지지하는 형태(`modifyDevice` 기본 throw). 구현 난이도 최소, 다만 "GPU insert"라는 목표 자체는 포기하는 선택 |

### 결정 5 — Search를 얼마나 "device-native"하게 만들 것인가

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. Persistent/hop-synchronized 커널 | 쿼리 배치를 GPU에 통째로 던지고 hop 진행까지 device 안에서 처리 | 진짜 `traverseDevice`. 구현 난이도·잠재 성능 모두 최대 |
| B. 얕은 오프로드 | 매 hop마다 host가 개입, 이웃 거리 계산만 GPU로 오프로드 | 구현 쉬움. 매 hop마다 host↔device 왕복 오버헤드로 그래프 탐색의 지연시간 이점이 상쇄될 수 있음 |
| C. 기존 GPU 그래프 라이브러리 호출 | CAGRA 등 기존 kernel을 그대로 사용 | 커널을 직접 짤 필요 없음. 대신 그래프를 해당 라이브러리가 요구하는 포맷(예: CSR)으로 맞춰야 함 — 결정 1-D와 사실상 연결된 선택 |

### 결정 6 — Visited-list를 GPU에서 어떻게 유지할 것인가

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. 쿼리 배치마다 전체 크기 비트셋 | `max_elements_` 크기 비트셋을 device global memory에 배치당 할당 | 단순, 정확. 배치가 크고 데이터셋이 크면 메모리 비용 커짐 |
| B. 고정 크기 해시셋/블룸필터 근사 | 메모리 절약 | false-positive로 인한 재방문 누락 가능 — recall에 영향, 검증 필요 |
| C. Warp-local 캐시 + global fallback | 최근 방문 노드만 빠른 캐시에 유지 | 구현 복잡도 증가, 메모리/정확도 절충의 중간 지점 |

---

## 4. 관찰 (결론이 아님)

- hnswlib은 host-only 라이브러리다 — `std::priority_queue`, `std::mutex`, `malloc`,
  함수 포인터 기반 `DISTFUNC`(`hnswlib.h:237-244`)를 사용하므로 **소스를 그대로 device에서
  컴파일/실행할 수 없다**. "포팅"은 항상 탐색 알고리즘의 재구현을 의미하며, hnswlib은 참고
  구현/정답지(ground truth) 역할에 머문다.
- 위 결정들 중 다수(2, 3, 4)는 서로 얽혀 있다 — 예를 들어 결정 4에서 "Insert는 Host 전용"을
  택하면 결정 3의 cross-region write 문제도 사실상 Host 동시성 모델(`link_list_locks_` 그대로
  재사용) 안에서 해결되므로 훨씬 단순해진다. 반대로 결정 1에서 D(CAGRA 대체)를 택하면 2/3/4는
  대부분 무의미해지고 결정 5/6만 남는다. 따라서 결정 1을 가장 먼저 확정하는 편이 나머지 논의
  범위를 크게 줄여준다.

---

> **아래 §5부터는 위 초안(§0-4)을 작성한 이후 진행된 후속 논의를 시간 순서대로 기록한 것이다.**
> §0-4는 최초 조사 스냅샷으로 그대로 남겨두고, 그 이후 구체적인 이식 방향을 좁혀가며 나온
> 결정/반박/재검토를 아래에 이어서 기록한다. 코드 구현은 아직 없다 — 전부 계획 단계다.

## 5. SVFusion 사례 검증 — 실제로 "HNSW를 GPU에서 실행"한 선례가 있는가

`references/svfusion`(VLDB 논문 "SVFusion: A CPU-GPU Co-Processing Architecture for
Large-Scale Real-Time Vector Search" + 소스 코드)를 근거로 확인했다.

**코드 근거**: `references/svfusion/lib/src/neighbors/`에는 `cagra*.cu`, `brute_force.cu`만
있고, `lib/src`/`examples` 전체에서 `hnswlib`을 include/링크하는 곳은 **한 곳도 없다**
(`grep -rl "hnswlib" lib/ examples/` 결과 없음). `lib/CMakeLists.txt`에도 참조가 없다.
`lib/cmake/patches/hnswlib.diff`가 존재하지만 내용은 전부 CPU 전용 수정이다 —
`enterpoint_node_` sentinel을 unsigned-safe하게 고치는 것, int8 거리함수 템플릿화,
`base_layer_only`/`num_seeds`(상위 레벨을 건너뛰고 균등 샘플 seed에서 바로 level 0로 진입하는
초대형 데이터셋용 트릭) 추가뿐이며 CUDA 코드는 없다.

**논문 근거**: Appendix B가 명시한다 — *"We implement SVFusion by extending the cuVS 24.12
library... We extensively refactor the **CAGRA** codebase—modifying tens of thousands of
lines—to support dynamic updates..."* SVFusion의 실제 엔진은 CAGRA를 대폭 수정한 것이며
HNSW가 아니다. §6.1에서 HNSW는 `M=48, ef_construction=ef_search=128`로 설정된 **비교
baseline**으로만 등장하고, 바로 이어 *"we create GPU-accelerated versions of FreshDiskANN
and HNSW **by offloading distance computations to GPU**"*라는 문장이 나온다 — 그래프
탐색(다음 노드 결정)은 그대로 CPU에 두고 거리 계산만 GPU로 오프로드한, 얕은 오프로드
baseline이라는 뜻이다. 그리고 이 baseline은 §6.2에서 스스로 나쁜 결과를 보고한다 —
*"GPU-accelerated baselines (i.e., FreshDiskANN(GPU) and HNSW(GPU)) perform 5.4-7.2×
**slower** than their CPU counterparts when datasets exceed GPU memory capacity."*

**결론**: SVFusion은 HNSW를 GPU에서 돌리는 사례가 아니다. 오히려 "거리 계산만 얕게
오프로드"하는 접근을 자기들이 직접 만들어 측정했고, 데이터가 GPU 메모리를 넘으면 CPU보다
느려진다는 걸 스스로 확인한 뒤 CAGRA 기반 자체 엔진으로 갔다. 이 레퍼런스 안에는 hnswlib의
그래프 control flow 자체를 GPU로 옮긴 선례가 없다 — 즉 §3 결정 1의 대안 D(CAGRA로 대체)가
실전에서 검증된 경로라는 근거는 되지만, 대안 A/B/C(hnswlib을 어떤 형태로든 GPU에 올리는 것)를
뒷받침할 선례는 없다.

---

## 6. hnswlib의 Traverse/Modify 분해 가능성

`addPoint()`(`hnswalg.h:1152-1266`)의 내부는 이미 두 단계로 나뉘어 있다.

1. **Traverse 단계** (`hnswalg.h:1240-1253`): 상위 레벨에서 진입점을 좁힌 뒤,
   `searchBaseLayer(currObj, data_point, level)`를 호출한다 — **검색(`searchKnn`)이 쓰는
   것과 완전히 동일한 함수**(`hnswalg.h:225-305`), 읽기 전용.
2. **Modify 단계** (`mutuallyConnectNewElement`, `hnswalg.h:506-630`): 1단계가 찾은 후보로
   새 노드의 링크리스트를 쓰고, 선택된 이웃들의 링크리스트까지 재배선한다.

이 구조는 Arachne의 `OpaqueData`/hint 설계(`index_adapter.hpp:18-29` — 예시로 정확히
*"graph neighbor candidates for HNSW-style insertion"*을 든다)와 `Controller::insert()`의
`Traverse -> hint 변환 -> Modify` 흐름(`controller.cpp:95-122`, `routeInsert()`가
`candidates.touched`/`candidates.hint`를 `ModifyRequest`로 그대로 이어줌)과 알고리즘
단위로는 잘 맞는다.

다만 **API 단위로는 안 나뉜다** — hnswlib은 이 두 단계를 쪼개 노출하는 public 메서드가 없다.
`addPoint()`는 모놀리식 호출이라, 어댑터가 `searchBaseLayer` 호출부와
`mutuallyConnectNewElement` 호출부를 직접 두 함수로 재조립해야 Traverse/Modify를 나눌 수
있다. 이걸 안 하면 대안은: Arachne가 요구하는 Traverse 단계에서 `searchKnn()`을 형식적으로
한 번 호출하고, Modify 단계에서 `addPoint()`를 통째로 호출 — `addPoint()`가 내부에서 또
탐색을 하므로 **탐색이 총 두 번** 일어나는 비효율을 감수하는 대신, hnswlib 소스는 한 글자도
안 건드리는 절충이 가능하다. (아래 §7에서 이 절충을 1차 이식 방향에 채택했다.)

또한 새 노드의 레벨이 기존 `maxlevel_`을 넘으면 `enterpoint_node_`/`maxlevel_`이라는
**전역** 상태가 바뀐다(`hnswalg.h:1261-1264`, `global` mutex로 보호) — 어느 Region에도
속하지 않는 인덱스 전체 메타데이터라서, Traverse/Modify 분리와 별개로 별도 처리가
필요하다.

---

## 7. 1차 이식 방향 — 합의된 단계별 계획

### 7.0 기각된 초안: "Region 1개 + 가짜 device stand-in"

최초 제안은 다음과 같았다 — (1) `data_level0_memory_` 전체를 Region 1개로만 등록, (2)
`traverseDevice()`는 `test/stress/stress_index.hpp`처럼 실제로는 host 계산을 그대로
수행하는 stand-in, (3) Insert는 `searchKnn()` 1회(형식적 Traverse) + `addPoint()`
통짜(Modify)로 hnswlib 무수정.

**기각 이유** (사용자 반박, 타당함):

1. Region을 1개로 묶으면 "전부 GPU에 있거나 전부 없거나"만 가능해져서, Arachne가 존재하는
   이유(데이터가 GPU보다 클 때 일부만 상주시키는 것) 자체를 테스트하지 못한다.
2. `traverseDevice()`가 실제로는 CPU 코드를 그대로 도는 stand-in이면, GpuOnly로 라우팅되든
   Hybrid로 라우팅되든 똑같은 코드가 돌아서 "GPU 상주 데이터로 실제 이득을 보는지"를 전혀
   검증하지 못한다 — 그럴 거면 애초에 Traverse/Modify를 나누고 GpuOnly 경로를 만들 이유가
   없다.
3. 종합하면 이건 "가장 어려운 문제를 피한다"기보다 사실상 "hnswlib과 실제로 얽히는 지점을
   전부 회피"하는 방향이었다.

이 기각을 계기로 아래 4단계 계획으로 재구성했다.

### 7.1 Phase 0 (Arachne 코어, hnsw 작업의 선행 조건) — host Modify가 Resident Region을 건드리면 무효화

**발견한 구체적 버그 가능성**: Insert가 Host 전용(Hybrid)으로 실행되면 `mutuallyConnectNewElement`가
host 메모리를 직접 고친다(새 노드 + 최대 M개 이웃). 이때 그 이웃이 마침 GPU에 승격돼
있는(Resident) Region에 속해 있으면, GPU 쪽 사본은 그대로인데 host만 바뀌는 불일치가
생긴다. `grep -rn ".modified" src/ include/` 결과 `ModifyResult::modified`(어떤 Region이
실제로 바뀌었는지 보고하는 필드, `index_adapter.hpp:75`)를 **읽는 코드가 어디에도 없다** —
`doc/code-documentation-v3.md` §28이 지적한 gap이 실제로 위험한 이유가 된 것이다.

구체적으로 두 가지 실패 모드가 있다 (`region_manager.cpp:1011-1053`,
`writeBackDirtyRegions()` 확인):

| 설정 | eviction 시 벌어지는 일 |
| --- | --- |
| `subregion_bytes = 0` (dirty-header 없음) | 무조건 dirty로 간주해 GPU→host로 write-back — **GPU의 옛날 데이터가 host의 새 insert 내용을 덮어써서 데이터 유실** |
| `subregion_bytes > 0` (dirty-header 있음) | host insert는 GPU dirty bit를 켤 수 없으므로(dirty bit는 GPU 커널의 `atomicOr`로만 켜짐) "clean"으로 오판, write-back을 건너뜀 — 이 경우는 덮어쓰기는 안 나지만, **eviction 전에 GPU가 그 Region을 읽으면(Phase 2에서 real kernel이 생기면) stale한 결과**가 나오는 문제가 남음 |

**필요한 작업**: `modifyHost()`가 반환한 `ModifyResult::modified`의 각 Region이 지금
`Resident`면, write-back 없이 강제로 `HostOnly`로 되돌리는 `RegionManager` 메서드를
추가하고 `Controller::dispatch(ModifyRequest)`에서 호출한다. host가 이미 최신 데이터를
갖고 있으므로 write-back은 불필요 — 그냥 무효화하면 다음에 필요할 때 자연히 재승격된다.
이건 hnsw 전용이 아니라 Modify를 host에서 실행하는 모든 adapter에 적용되는 일반 배선이라,
seq 어댑터에도 나중에 필요할 가능성이 높다.

### 7.2 Phase 1 — Region 분할 + Host 경로로 배선 검증 (GPU 커널 아직 없음)

- `HnswIndex : IAdapter`, `HnswRegion : IRegion` — `data_level0_memory_`를 `StressIndex`
  방식대로 internal id 구간별로 슬라이스, `subregion_bytes = size_data_per_element_`.
- capacity 고정, `resizeIndex()` 호출 금지 (`hnswalg.h:633-656`의 `realloc`이 GPU-promoted
  상태의 host 포인터를 무효화할 수 있음).
- `traverseHost()` = `searchKnn()`, insert = `addPoint()`(§6에서 정리한 "이중 탐색" 절충
  그대로), delete = `markDelete()`.
- **추가로 발견한 gap**: `TraverseResult::touched`를 채우려면 탐색 도중 실제로 방문한 노드
  id 목록이 필요한데, `searchKnn()`/`searchBaseLayerST`는 이걸 공개 API로 반환하지 않는다
  (최종 top-k만 반환). 두 가지 중 결정 필요:
  - (a) top-k 결과의 Region만 `touched`로 근사 — 구현 간단하지만 중간 hop에서 지나친
    Region은 계속 promotion 후보에서 빠짐.
  - (b) hnswlib에 작은 패치(방문 id를 모으는 출력 파라미터 추가) — SVFusion의
    `hnswlib.diff`도 이 정도 국소 패치였다는 선례가 있어, (b) 쪽을 권장.
- Insert의 `ModifyResult::modified` 계산: `addPoint()` 완료 후 새 노드의 최종 level-0
  링크리스트(`get_linklist0`)를 읽어 "새 노드 + 그 이웃들"의 Region을 modified로 보수적으로
  보고.
- Phase 0의 무효화 로직을 이 modified footprint로 연결.
- 검증: GPU 예산을 작게 잡고 stage2/stage3류 stress 테스트 패턴을 재현해, 승격/축출이
  반복되는 와중에도 recall이 깨지지 않는지 확인.

### 7.3 Phase 2 — Search에 진짜(얕은 오프로드) GPU 커널

- `traverseDevice()`: "다음 노드 결정"은 host에 남기고, "현재 프론티어의 이웃들과의 거리
  계산"만 실제 CUDA 커널로 GPU에서 수행 — §5에서 확인한 SVFusion의 HNSW(GPU) baseline과
  같은 구조.
- 실행 조건: Anchor의 `residency_hints`가 전부 pin에 성공하면 GpuOnly 시도, 실패하면
  Arachne가 이미 가진 fallback 경로(`Controller::routeSearch()`의
  `completed_within_scope`/`fallback_to_hybrid`, `controller.cpp:71-76`)를 그대로 재사용해
  전체를 Hybrid로 재시도.
- 검증 목표는 "빨라지는지"가 아니라 **"빨라지는지 실측"**이다 — SVFusion 자신도 이 방식이
  데이터가 GPU 메모리를 넘으면 5.4~7.2배 느려진다고 보고했으므로, 우리 쪽에서도 같은 현상이
  나올 수 있다는 전제로 측정해야 한다. (§9에서 이 커널 설계를 더 구체화했다.)

### 7.4 Phase 3 (보류) — Insert의 cross-region 쓰기 문제

§3 결정 3(cross-region write)을 그대로 보류한다. `addPoint()`를 진짜로 Traverse/Modify
두 함수로 쪼개 이중 탐색 비용을 없애는 작업도 여기 포함된다. Phase 1/2가 실제로 잘 도는지
보고 나서 착수 여부를 결정한다.

---

## 8. 축소판 — Modify를 아예 비워둔 정적(read-only) 인덱스

Insert/Delete를 전혀 지원하지 않는(그래프를 Arachne 밖에서 미리 다 만들어놓고 이후로는
검색만 하는) 훨씬 단순한 시작점도 검토했다.

**사라지는 것**: Phase 0(host가 GPU-resident Region을 몰래 고치는 문제)과 Phase 3(insert의
cross-region 쓰기)은 인덱스가 만들어진 뒤로 아무도 host 데이터를 안 고치므로 **원인 자체가
성립하지 않는다.**

**남는 것**: Region 분할(§7.2)은 그대로 필요하고 난이도도 동일하다. "탐색이 어떤 노드를
방문했는지 hnswlib이 노출 안 한다"는 gap(§7.2의 (a)/(b) 결정)도 Search 자체의 문제라
Modify 유무와 무관하게 남는다. 진짜 GPU 커널(§7.3)을 넣을지도 여전히 독립적인 선택지다.

이 축소판으로도 Region 승격/축출, generation 검증, RoutingCache가 Anchor를 기억했다가
GpuOnly로 재시도하는 것까지 — Arachne 핵심 기계장치는 검색만으로 전부 실제 hnswlib 데이터로
exercise할 수 있다(`Controller::search()` 안에서 promotion 후보 등록/RoutingCache 갱신이
전부 검색 경로로만 일어나기 때문에, insert 없이도 검증 가능).

capacity 고정/`resizeIndex()` 금지 제약도 정적 인덱스에서는 애초에 자동으로 성립한다.

---

## 9. GPU 커널 설계 시 확정해야 할 요구사항

Phase 2(§7.3)의 커널을 실제로 설계할 때 짚어야 할 세 가지를 논의했다.

### 9.1 배치 지원

`IAdapter::traverseDevice(const std::vector<TraverseRequest>&)`부터 이미 배치 형태다
(`index_adapter.hpp:145`). 관건은 이 vector를 커널 안에서 **진짜로 한꺼번에 병렬 처리**하느냐,
아니면 host 루프로 쿼리 하나씩 작은 커널을 여러 번 launch하느냐다 — 후자는 GPU를 거의 못
채운다. 배치 안의 쿼리들은 그래프에서 몇 hop을 가야 할지 서로 다르므로, **hop-synchronized
커널**(SVFusion 논문 Appendix Algorithm 4, CAGRA 방식 — 매 반복마다 배치 내 모든 쿼리를
동시에 한 스텝씩 전진, 먼저 수렴한 쿼리는 이후 반복에서 idle)이 표준적인 구조다.
`OpScheduler`의 `traverse_batch_size`(`op_scheduler.hpp`)가 이미 요청을 모아 배치로
넘기는 장치이므로, 커널을 이 배치 크기에 맞춰 처리하도록 짜는 게 자연스럽다.

### 9.2 Cross-region(GPU에 없는 이웃) 처리 — 두 가지 안을 검토, (B) + 개선안으로 결정

처음 제안("GPU에 없으면 그냥 무시하고 GPU에 있는 것만으로 계속 진행")은 **반박했다** —
brute-force와 달리 HNSW는 그래프라서, 다음 노드로 갈 유일한 경로가 하필 GPU에 없는
hub 노드를 거쳐야 하는 경우, 그 방향 자체를 영영 놓친다. 그리고 어떤 노드가 GPU에 있는지는
Arachne의 승격/축출 타이밍에 달려 있어 쿼리·그래프 구조와 무관하므로, 이렇게 생기는 recall
손실은 예측·재현이 어렵다. Arachne는 이미 이 상황에 대한 답(`completed_within_scope` +
`fallback_to_hybrid`로 전체를 host 재시도)을 갖고 있고, "무시"는 이 안전장치를 없애고
덜 정확한 답을 최종 답으로 받아들이자는 것과 같다.

이후 두 가지 정제안으로 나눠 논의했다:

- **(A) hop마다 즉시 병합**: 매 스텝마다 GPU-resident 이웃은 GPU가, 아닌 이웃은 CPU가
  나눠 계산하고 합친 뒤 다음 노드를 결정 — SVFusion 논문 Algorithm 1(`D_CPU ∪ D_GPU`로
  후보 큐 갱신)과 사실상 동일한 구조. recall 문제는 진짜로 해결되지만(hub를 넘어 계속
  탐색 가능), 매 hop마다 host 개입이 필요해 **배치 처리 시 stall이 커진다.**
- **(B) 사후 remainder 병합**: GPU가 resident 데이터만으로 끝까지(여러 hop) 탐색을 마치고,
  "이런 노드들을 못 봤다"는 목록을 들고 돌아오면, host가 배치 전체에 대해 **한 번만** 그
  목록을 처리해서 병합. 사용자가 이 방향을 선택했다 — 이유: 배치 단위로 넘기는데 매
  hop마다 fallback하면 stall이 많아지기 때문.

**(B)를 선택한 것에 동의한다** — 처리량 관점에서 타당하다. 다만 (B)는 "놓친 노드 자체의
거리"만 평가해 후보에 추가하는 것으로는, 그 노드의 **이웃들**(진짜 정답이 있었을 곳)을
여전히 아무도 안 가보는 문제가 남는다 — hub 하나를 후보로 넣어주는 것뿐, 그 너머로 이어서
탐색하는 건 아니기 때문이다. **개선안으로 제안한 것**: remainder 단계를 "놓친 노드의 거리
하나만 계산"이 아니라 "놓친 노드에서 시작해 host가 짧게(예: 1~2 hop) 이어서 걸어본 뒤 병합"으로
확장. 여전히 host 왕복은 배치당 한 번(remainder 단계 전체가 한 덩어리)이라 stall 문제는
피하면서, hub를 건너뛴 손실을 부분적으로 메꿀 수 있다. 몇 hop까지 이어서 걸을지는 recall과
remainder 계산 비용의 트레이드오프라 실측이 필요하다.

### 9.3 커널 내부에서 Region 주소를 어떻게 찾을 것인가

Arachne는 host 레벨에서 이미 이 문제를 풀어놨다 — `DeviceRegionHandle`은 원시 포인터가
아니라 `{uint64_t id}`뿐인 opaque 값이고(`device_region_handle.hpp`), 실제 포인터는
`DeviceRegionPool::acquire()`로 얻는 `Lease`의 생존 기간 동안만 존재한다(Region이
compaction/eviction으로 옮겨지거나 사라질 수 있기 때문). 이 패턴을 커널 내부로 확장하면
된다:

1. 커널 launch **직전**, host가 이번 배치가 필요로 할 만한 Region들(각 Anchor의
   residency_hints) 전부에 대해 `Lease`를 미리 확보한다.
2. `{region_id → 지금 이 순간의 device 포인터}` 조회 테이블을 만들어 커널 인자로 넘긴다.
3. 커널 안에서 "지금 노드가 몇 번 Region인지"는 계산으로 나오므로(`internal_id /
   vectors_per_region`), 그 id로 조회 테이블을 찾아 실제 주소를 얻는다 — 그래프 안에 주소를
   하드코딩하지 않는다.
4. 커널이 끝날 때까지 이 Lease들을 계속 쥐고 있어야 한다 — Arachne의 기존 불변조건("pin/lease가
   남은 Region은 축출/재배치 안 함")이 이걸 그대로 보장하므로 새 안전장치는 필요 없고, 커널
   실행 시간 전체로 Lease의 생존 범위만 늘리면 된다.

### 9.4 §9.2와 §9.3은 같은 메커니즘을 공유한다

§9.3에서 "region id → 지금 이 순간의 포인터" 테이블을 만드는 바로 그 순간에, "이 Region이
지금 resident인지"도 같이 알게 된다 — 즉 §9.2의 "어떤 이웃이 접근 가능한지" 판단은 §9.3
테이블을 만드는 과정에서 자연히 나오는 부산물이다. 이 정보를 "누락되면 조용히 무시"할지
"누락 목록으로 모아 remainder 단계로 넘길지"만 다르게 쓰면 되므로, 구현 난이도 차이는 크지
않다.

---

## 10. Entry point 재사용 최적화

### 10.1 발상

hnswlib 검색은 매번 전역 `enterpoint_node_`에서 출발해 상위 레벨을 하나씩 내려가며 쿼리에
가까운 지점을 찾는 사전 작업을 한다(`hnswalg.h:1277-1302`, `searchKnn`). 이건 순차적이고
작은 계산이라 GPU로 병렬화하기도 애매하다. GpuOnly로 라우팅되는 순간 Arachne는 이미
RoutingCache를 통해 "이 쿼리는 전에 봤던 어떤 위치 근처다"라는 판단을 끝낸 상태이므로, 이
상위 레벨 하강을 건너뛰고 그 지점에서 바로 level 0 탐색을 시작하자는 아이디어.

### 10.2 첫 시도(Anchor를 그 자체로 취급)가 왜 깨지는지

"hnswlib의 entry point 자체를 Arachne의 (유일한) Anchor로 등록하면 어떨까"로 처음
해석했으나, 이건 §7.0에서 기각한 "Region 1개" 문제가 Anchor 레벨에서 재현되는 결과를
낳는다. `RegionManager`의 축출 단위는 Region이 아니라 **Anchor**다 —
`retireAnchorsNow()`(`region_manager.cpp:869-888`)와 `releaseAnchor()`(`region_manager.cpp:281-319`)
모두 `forget(anchor_id)`로 "그 Anchor 말고 아무도 안 쓰는 Region"만 orphan으로 골라
반환한다(`removeDependency()`가 "마지막 Anchor가 사라졌을 때만" true를 반환,
`region_manager.hpp:248-252`). 검색을 시작할 때마다 항상 같은 entry point에서 출발하는
hnswlib의 특성상, 이걸 유일한 Anchor로 등록하면 서로 다른 위치를 찾는 모든 쿼리가 같은
Anchor 아래 쌓이게 되고, 그 Anchor의 dependency 집합은 사실상 전체 데이터로 불어난다 —
Region 하나가 개별적으로 반환되려면 "마지막 Anchor가 사라져야" 하는데 Anchor가 이거
하나뿐이니, 이 Anchor를 축출하면 전체 데이터가 한 번에 다 축출되는 것과 같아진다.

### 10.3 정정된 의도와 Region-키 시도, 그리고 그 함정

사용자가 정정한 실제 의도: entry point 자체를 Anchor로 만들자는 게 아니라, "이미 RoutingCache가
찾아준 근처 지점을, hnswlib이 매번 새로 하는 상위 레벨 하강 대신 재사용하자"는 것이었다.

이를 구현하려면 adapter가 "이 요청이 어떤 Anchor 때문에 왔는지" 알아야 하는데,
`TraverseRequest`(`index_adapter.hpp:34-43`)엔 `anchor_id` 필드가 없다 — Anchor id는
Controller/RegionManager끼리만 주고받는 값이다. 이걸 피하려고 **Region을 키로 쓰는 방안**
(adapter가 자체적으로 `RegionId → 좋은 진입 노드` 캐시를 두고, Hybrid 탐색 성공 시 갱신,
GpuOnly 시 조회)을 제안했으나, 이 과정에서 더 근본적인 문제를 발견했다:

hnswlib의 internal id는 **삽입 순서**로 부여된다(`addPoint()`,
`hnswalg.h:1180-1182`: `cur_c = cur_element_count; cur_element_count++;`) — 공간적
위치와 아무 관계가 없다. 그런데 §7.2에서 채택한 Region 분할 방식은 정확히 이 id를
연속 구간으로 자른다. 즉 "internal id가 이웃"이라는 게 "그래프에서 이웃"이라는 뜻이 전혀
아니라서, 한 Region 안의 벡터들이 실제로는 서로 무관한 위치에 흩어져 있을 수 있고, 반대로
쿼리 하나의 탐색 경로가 지나가는 노드들은 id 공간 전체에 흩어져 있어 여러 Region에
분산된다. 이 때문에 "Region 하나에 서로 무관한 여러 Anchor가 몰리는 것"은 예외가 아니라
**당연한 결과**이고, "Region당 진입점 하나"는 서로 다른 지역을 대표하는 값 하나를 억지로
캐싱하는 꼴이 되어 의미가 약해진다.

(참고: `StressIndex`의 brute-force에서는 id 순서가 아무 의미가 없어도 상관없었다 — 모든
벡터를 순서 무관하게 전부 스캔하기 때문. 그래프 탐색에서는 다르다 — Region 승격이
도움이 되려면 그 Region이 실제 지역을 이뤄야 하는데, id 순서 기반 분할은 이를 보장 못한다.)

이 문제 자체의 해결 방향(참고용, §7.2 결정과 별개로 열려 있음): §8에서 이미 "Modify는
비워두고 hnswlib 그래프를 Arachne 밖에서 미리 다 만든다"고 정했으므로, 삽입 순서를 우리가
통제할 수 있다 — 클러스터링(예: k-means) 후 클러스터 단위로 삽입하거나, 더 가볍게는
공간을 보존하는 정렬(예: 축 기준 정렬, space-filling curve)로 삽입 순서를 정하면 id 연속
구간이 실제 지역과 맞아떨어지게 만들 수 있다. 다만 이건 추가 전처리 단계가 필요한 별도
결정 사항이다.

### 10.4 최종 방향 — Anchor에 직접 1:1로 기록 (Region 우회)

Region을 키로 쓰려 했던 건 "Arachne core를 최대한 안 건드린다"는 자체 제약 때문이었는데,
이 제약이 불필요했다는 지적을 받아들였다. Anchor는 애초에 쿼리 벡터 하나에 대응하는 고유
식별자이므로, **Anchor id → 진입 노드**를 1:1로 직접 기록하면 여러 Anchor가 섞이는 문제
자체가 생기지 않는다(§10.3의 Region 지역성 문제와 무관해진다).

구체적 설계:

1. `TraverseRequest`(또는 `Controller::dispatch()`가 이미 별도 인자로 들고 있는
   `promotion_anchor_id`를 요청에 실어 넘기는 방식)에 `anchor_id`를 추가하는 **작은
   core 변경**이 필요하다 — Phase 0(§7.1)에서 이미 받아들인 것과 같은 성격의 변경이라
   특별히 회피할 이유가 없었다.
2. adapter는 내부에 `unordered_map<VectorId /*anchor_id*/, tableint>`를 둔다.
3. Hybrid 탐색이 성공적으로 끝날 때마다(=`traverseHost()`가 상위 레벨 하강 + level 0
   탐색을 실제로 수행했을 때), 최종적으로 도달한 실제 노드의 internal id를 그 Anchor id로
   기록해둔다.
4. 이후 같은 Anchor가 GpuOnly로 재시도될 때, adapter는 이 표에서 진입점을 찾아 상위 레벨
   하강을 건너뛰고 바로 그 지점에서 level 0 탐색을 시작한다.

정적 인덱스(§8) 전제 하에서는 그래프가 안 바뀌므로 이 캐시가 stale해질 걱정도 없다. 다만
캐시된 노드가 속한 Region이 그새 축출됐다면, 기존 residency-hint/pin 검증 메커니즘이
독립적으로 이를 걸러내므로 새로운 정확성 위험은 추가되지 않는다.

### 10.5 별도로 남은 문제

§10.3에서 발견한 "Region(id 연속 구간)이 그래프 지역성과 안 맞는다"는 사실 자체는 여전히
유효하고, **승격/축출이 Region 단위로 일어나는 것의 전반적인 효율**에는 계속 영향을 주는
문제로 남는다 — 다만 이건 §10.4의 entry point 캐싱 문제와는 무관해졌으므로, 별도 항목으로
분리해 나중에 다룬다(§7.2 결정 (a)/(b), 그리고 Region 분할 기준 자체의 재검토와 함께).

---

## 11. 종합 — 현재까지 정리된 상태

| 항목 | 상태 |
| --- | --- |
| hnswlib을 그대로 GPU에서 실행 가능한가 (§0, §4, §5) | 불가능 — host-only 라이브러리. SVFusion도 실제로는 CAGRA로 감. "포팅"은 항상 재구현을 의미 |
| Traverse/Modify가 hnswlib에 자연스럽게 대응되는가 (§6) | 알고리즘적으로는 대응(`addPoint` 내부가 이미 그 구조), API로는 안 나뉨 — 이중 탐색을 감수하고 wrap하는 절충 채택 |
| Region 분할 단위 | internal id 연속 구간(§7.2) 채택, 단 그래프 지역성과 불일치하는 문제 발견(§10.3) — 별도 열린 문제로 분리 |
| Insert(Modify) 처리 | 1차: Host 전용 고정(§7.1/7.4), cross-region 쓰기는 보류. 더 단순화하면 아예 Modify 비활성(정적 인덱스, §8)도 가능 |
| GPU 커널이 실제로 필요한가 | Phase 1(§7.2)은 GPU 커널 없이(host-only) 배선만 검증. Phase 2(§7.3, §9)에서 진짜(얕은 오프로드) 커널 추가 — 배치 처리(§9.1), cross-region 처리는 (B) 사후 remainder + 짧은 재탐색 개선안(§9.2), 커널 내부 주소 해석은 opaque id + lease + 포인터 테이블 확장(§9.3) |
| Entry point 최적화 | Anchor id를 `TraverseRequest`에 실어 넘기는 작은 core 변경 + adapter 내부 `anchor_id → tableint` 캐시로 결정(§10.4) |
| 아직 결정 안 된 것 | (a) 방문 노드 목록 노출을 위한 hnswlib 패치 여부(§7.2), (b) Region 분할과 그래프 지역성을 맞추기 위한 삽입 순서 전처리 방식(§10.3/10.5), (c) 정적 인덱스(§8) vs 전체 파이프라인(§7) 중 실제로 먼저 구현할 범위 |
