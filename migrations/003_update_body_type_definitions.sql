-- 체형별 정의 데이터 업데이트
-- 사용 불가능한 드레스 스타일(드롭 웨이스트, 하이웨이스트, 엠파이어 라인)을 보유한 드레스로 수정
-- 실행 방법: mysql -u [사용자명] -p [데이터베이스명] < migrations/003_update_body_type_definitions.sql

-- 1. 키가 작은 체형: 엠파이어 라인 → 프린세스
UPDATE body_type_definitions 
SET 
    strengths = '작은 키의 신부님은 허리선이 높게 올라간 프린세스 드레스가 체형을 길어 보이게 만들어줍니다.',
    recommended_dresses = '프린세스',
    avoid_dresses = '슬림 (키가 작아 보일 수 있음)'
WHERE body_feature = '키가 작은 체형';

-- 2. 허리가 짧은 체형: 드롭 웨이스트 → 벨라인, 하이웨이스트 제거
UPDATE body_type_definitions 
SET 
    strengths = '허리선을 강조하는 벨라인 드레스는 하체를 길어 보이게 만들어 전체적인 비율을 맞춰줍니다.',
    style_tips = '허리 라인을 강조하여 비율을 조화롭게 연출하는 스타일이 적합합니다.',
    recommended_dresses = '벨라인',
    avoid_dresses = '슬림 (허리가 더 짧아 보일 수 있음)'
WHERE body_feature = '허리가 짧은 체형';

-- 3. 키가 큰 체형: 엠파이어 라인 제거, 미니드레스 추가
UPDATE body_type_definitions 
SET 
    recommended_dresses = '슬림, 미니드레스',
    avoid_dresses = '프린세스 (키가 더 커 보일 수 있음)'
WHERE body_feature = '키가 큰 체형';

-- 업데이트 확인
SELECT body_feature, recommended_dresses, avoid_dresses 
FROM body_type_definitions 
WHERE body_feature IN ('키가 작은 체형', '허리가 짧은 체형', '키가 큰 체형')
ORDER BY body_feature;

