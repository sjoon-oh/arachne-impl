# Exhaustive Scan(Sequential/Brute-force) GPU 포팅 고려사항 (draft)

> 작성: 2026-08-07
> 범위: 그래프/인덱스 구조 없이 매 쿼리마다 전체 데이터셋을 스캔하는 brute-force 어댑터
> (`cpp/index/seq` 아래에 C++로 구현 예정)를 GPU Region 모델 위에 올리기 위한 사전 조사.
> 코드 구현은 포함하지 않는다.
>
> **범위 밖 명시**: `thirdparty/hnswlib`도 `bruteforce.h`에 자체 brute-force 구현을 갖고
> 있지만, 그건 hnswlib 자체 API(`SpaceInterface`/`searchKnn`)에 묶여 있어 Arachne의
> `IAdapter`/`IRegion` 계약과 직접 맞지 않는다. 여기서는 참고 자료로만 사용한다.

---

## 0. 왜 hnsw보다 먼저 볼 가치가 있는가

`doc/code-documentation-v3.md` §28은 "실제 production GPU-native ANN adapter와 kernel"의
부재를 현재 구현의 경계로 명시한다. 이미 `test/stress/stress_index.hpp`/`.cpp`가 정확히
brute-force 방식의 test-double(`StressIndex`)로 Arachne Region 모델에 맞춰 조정돼 있으므로,
이 폴더는 그것을 **실제 후보로 formalize하는 가장 낮은 리스크 경로**로 볼 수 있다 —
그래프 엣지가 없어 `hnsw/report.md`에서 다룬 cross-region write 같은 문제 자체가 발생하지
않는다.

---

## 1. 참고 구조 요약 (근거 코드)

| 소스 | 레이아웃 |
| --- | --- |
| `thirdparty/hnswlib/hnswlib/bruteforce.h` | 연속 `data_` slab (`size_per_element_ = data_size_ + sizeof(labeltype)`, `bruteforce.h:51-52`). `addPoint`=append (`bruteforce.h:64-83`), `removePoint`=**swap-with-last** (`bruteforce.h:86-103`, 마지막 원소를 삭제된 슬롯으로 옮기고 `cur_element_count--`), `searchKnn`=선형 스캔 + partial top-k (`bruteforce.h:106-125`). 그래프도, 포인터 체이싱도 없음 |
| `test/stress/stress_index.hpp`/`.cpp` | 이미 Arachne Region 모델에 맞춰 조정된 버전. `buffer_`를 `vectors_per_region`개씩 등분해 슬라이스마다 하나의 `StressRegion`(`IRegion`) 등록 (`stress_index.hpp:14-34` 주석에 레이아웃 그림 포함). `subregion_bytes` = 벡터 1개 크기, `id_to_slot_`로 VectorId→슬롯 매핑, 삭제는 `deleted_`에 **tombstone만 하고 슬롯은 재활용하지 않음**(`stress_index.hpp:30-34` 주석이 이 트레이드오프를 직접 명시) |
| `StressIndex::DistanceSquared` | `test/stress/stress_index.cpp:101-109` | 차원별 raw scalar loop — `util/distance.hpp`의 Highway SIMD 경로를 쓰지 않는 순수 참고 구현 |
| `StressIndex::traverseDevice`/`modifyDevice` | `stress_index.hpp:101-113` | **실제 GPU 커널이 아니라 host buffer를 그대로 재사용하는 stage 1-3용 stand-in** — 주석이 "until stage 4 adds a real write kernel"이라고 명시. 즉 지금 이 경로도 아직 device에서 실제로 계산하지 않는다 |

---

## 2. 가능한 부분

- **Region 경계 설계가 사실상 이미 풀려 있다.** `StressIndex`의 `vectors_per_region` 슬라이싱
  방식을 거의 그대로 채택할 수 있다 — 그래프 엣지가 없으므로 삽입 1건이 자기 슬롯이 속한
  Region 하나만 건드리고 끝난다. `hnsw/report.md` 결정 3에서 다룬 cross-region write 문제가
  구조적으로 발생하지 않는다.
- **거리 계산은 전형적인 embarrassingly-parallel dense 연산**이다 — 쿼리×후보 쌍마다 완전히
  독립적이므로, 배치 쿼리(Nq) × 전체 후보(Nc)의 거리 행렬 계산은 GPU가 가장 잘하는 작업
  형태다. L2 거리는 `‖a‖² + ‖b‖² − 2·a·b`로 전개하면 GEMM으로도 재구성 가능한 표준 커널이다.
- **`raft::raft`가 이미 하드 의존성으로 링크돼 있다** (`CMakeLists.txt`의
  `find_package(raft CONFIG REQUIRED)` + `target_link_libraries(arachne_core PUBLIC
  raft::raft)`). RAFT/cuVS가 제공하는 brute-force 최근접 이웃 primitive를 재사용하는 경로가
  이미 열려 있어, 커널을 처음부터 짤 필요가 없을 수도 있다.
- **Host 경로부터도 개선 여지가 확인됐다** — 현재 `StressIndex::DistanceSquared`는 SIMD 없이
  raw loop을 쓴다(위 표 참고). `util/distance.hpp`(Highway 기반, `ASRoutingCacheHnsw`의 cosine
  정규화가 이미 사용 중)를 `traverseHost`에 재사용하면 GPU 작업과 무관하게 Host 경로 성능부터
  올릴 수 있다.

---

## 3. 반드시 결정이 필요한 부분과 대안

### 결정 1 — distance/top-k를 무엇으로 구현할 것인가

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. RAFT/cuVS의 brute-force primitive 재사용 | 기존 의존성을 그대로 활용 | 구현량 최소. RAFT API 버전/시그니처에 맞춰야 하는 결합도 발생 |
| B. 직접 CUDA 커널 (tile 기반) | query-batch × candidate-tile 형태로 직접 작성 | Arachne의 dirty-header/Region 크기와 커널 tile을 맞춰 최적화 가능. 구현/검증 비용 최대 |
| C. cuBLAS GEMM 기반 거리 행렬 + 별도 top-k 커널 | L2를 노름 전개로 GEMM화 | 고차원·대량 배치에서 유리할 수 있음. 정밀도(부동소수 상쇄 오차) 고려 필요, top-k는 별도로 필요 |

### 결정 2 — top-k selection을 어디서 수행할 것인가

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. GPU에서 전부 처리 | distance + top-k 커널까지 device에서, host round-trip 최소화 | PCIe 전송량 최소, 구현 복잡도 증가 (정렬/heap 커널 필요) |
| B. GPU는 distance만, top-k는 host | 거리 값만 device→host로 가져와 `std::partial_sort` 등 재사용 | 구현 단순(현재 `StressIndex`와 같은 top-k 로직 재사용 가능). 거리 행렬 전체를 host로 옮기므로 배치가 크면 전송 비용 증가 |

### 결정 3 — Region 분할 단위

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. 고정 vector 개수 (`vectors_per_region`) | `StressIndex`와 동일 | 기존 test-double과 개념적으로 바로 호환, 검증된 패턴 |
| B. 고정 byte 크기 | `UnitPoolArena`의 `unit_bytes`에 맞춤 | Pooled 모드의 near-fit reuse(§20, `doc/code-documentation-v3.md`)와 정합성이 더 좋을 가능성 — 단, `dim`/`dtype`이 섞인 워크로드에서는 vector 개수가 Region마다 달라짐 |
| C. GPU 예산에 맞춘 동적 크기 | Region 하나 = "한 번의 스캔 타일" | 커널 tile 크기와 직접 연동 가능. Region 경계가 워크로드/GPU 예산에 따라 달라져 재현성・디버깅이 어려워질 수 있음 |

### 결정 4 — Delete를 어떻게 처리할 것인가

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. Swap-with-last | `hnswlib::BruteforceSearch`와 동일(`bruteforce.h:86-103`) — 즉시 공간 회수 | 공간 효율 좋음. 삭제된 슬롯에 마지막 원소가 옮겨오면서 **다른 VectorId의 내부 슬롯 위치가 바뀜** → `id_to_slot_`뿐 아니라 그 슬롯이 속한 Region의 dirty/lease 상태까지 갱신해야 함 — GPU promote된 Region이라면 D2H/H2D 재조정 비용 발생 |
| B. Tombstone, 재활용 안 함 | `StressIndex`와 동일(`stress_index.hpp:30-34`) | 구현 가장 단순, 슬롯 위치 불변이라 Region 상태 갱신 없음. `StressIndex` 주석이 이미 명시하듯 heavy delete/reinsert churn에서 결국 `capacity` 소진 위험 |
| C. Tombstone + background compaction | `ASRoutingCache`의 tombstone-ratio-triggered 압축 패턴(§7, 이전 대화에서 확인한 `as_routing_cache.cpp` 구조) 재사용 | 아키텍처 일관성(이미 검증된 패턴 재사용). 다만 여긴 라우팅 캐시가 아니라 실 벡터 payload이므로, 압축 1회의 비용이 훨씬 큼(전체 벡터 데이터 재배치, GPU 쪽 Region까지 동반 이동) |

### 결정 5 — dirty-tracking을 얼마나 세밀하게 가져갈 것인가

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. `subregion_bytes` = 벡터 1개 | `StressIndex`와 동일 | insert 시 정확히 그 벡터만 dirty로 표시 — write-back 비용 최소화 |
| B. `subregion_bytes` = 0 (Region 전체를 하나의 dirty 단위로) | 세밀한 추적 생략 | 구현/오버헤드 단순화. brute-force는 insert가 대부분 append라 세밀한 추적의 이득이 애초에 크지 않을 수 있음 — 다만 결정 4-A(swap-with-last)를 택하면 두 슬롯이 동시에 dirty가 되므로 A쪽이 더 유리해짐 |

### 결정 6 — Insert를 GpuOnly로 허용할 것인가

| 대안 | 설명 | 트레이드오프 |
| --- | --- | --- |
| A. 항상 Hybrid | host가 append 후 dirty 표시만, GPU 반영은 다음 promotion/write-back에 위임 | 구현 단순, `hnsw` 쪽 결정 4-C와 같은 보수적 선택 |
| B. GpuOnly 허용 (`modifyDevice` 구현) | 이미 promote된 Region에 대해 GPU에서 직접 append | append 자체는 hnsw의 이웃 재배선보다 훨씬 쉬움 — 끝에 하나 추가하는 연산이라 `atomicAdd` 기반 인덱스 증가만으로 충분할 가능성이 높음. 그래도 Region 경계(꽉 찬 Region 처리, 새 Region 필요 시점)를 device 쪽에서 어떻게 판단할지는 별도 결정 필요 |

---

## 4. 관찰 (결론이 아님)

- `hnsw/report.md`와 비교하면, 여기서 다룬 결정들은 서로 상당히 독립적이다 — 그래프가 없어서
  하나의 선택이 다른 결정의 전제를 바꾸는 경우가 적다(`hnsw` 쪽 결정 1이 나머지를 좌우하는
  것과 대조적).
- `StressIndex::traverseDevice`/`modifyDevice`가 아직 host 버퍼를 재사용하는 stand-in이라는
  점(§1)은, "GPU 커널이 Arachne의 lease/generation/dirty 모델과 실제로 맞물려 도는지" 자체가
  이 레포 어디서도 검증된 적이 없다는 뜻이다. 그래프 탐색의 알고리즘적 난제(동시성, cross-region
  write)가 없는 이 brute-force 경로가, 그 배선 자체를 먼저 검증하는 데는 더 낮은 리스크의
  시작점일 수 있다 — 다만 이건 순서에 대한 관찰이지 결정은 아니다.
