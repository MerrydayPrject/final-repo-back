-- 체형별 정의 테이블 생성 및 초기 데이터 삽입
-- 실행 방법: mysql -u [사용자명] -p [데이터베이스명] < migrations/001_body_type_definitions.sql

-- 테이블 생성
CREATE TABLE IF NOT EXISTS body_type_definitions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    body_feature VARCHAR(50) NOT NULL UNIQUE COMMENT '체형 특징 (키가 작은 체형, 글래머러스한 체형 등)',
    strengths TEXT COMMENT '체형의 강점 설명',
    style_tips TEXT COMMENT '스타일 팁',
    recommended_dresses TEXT COMMENT '추천 드레스 스타일',
    avoid_dresses TEXT COMMENT '피해야 할 드레스 스타일',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_body_feature (body_feature)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='체형별 정의 및 드레스 추천 정보';

-- 기존 데이터 삭제 (재실행 시)
DELETE FROM body_type_definitions;

-- 1. 키가 작은 체형
INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses) VALUES
('키가 작은 체형',
'작은 키의 신부님은 허리선이 높게 올라간 엠파이어 라인 드레스가 체형을 길어 보이게 만들어줍니다.',
'심플한 디자인을 선택하면 키가 더 커 보이는 효과를 볼 수 있습니다.',
'엠파이어 라인',
'긴 기장의 드레스 (키가 작아 보일 수 있음)');

-- 2. 글래머러스한 체형
INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses) VALUES
('글래머러스한 체형',
'곡선미가 뚜렷한 체형으로, 몸매 라인을 강조하면서도 우아한 분위기를 연출할 수 있습니다.',
'머메이드 라인 드레스가 제격입니다.',
'머메이드',
'슬림 (곡선미를 제대로 살리기 어려움)');

-- 3. 어깨가 넓은 체형
INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses) VALUES
('어깨가 넓은 체형',
'어깨가 넓다면 상체는 비교적 심플하게 정리되고, 스커트가 자연스럽게 퍼지는 A라인이나 프린세스 라인이 균형 잡힌 실루엣을 만들어줍니다.',
'상체를 심플하게 정리하고 하체 볼륨을 주는 스타일이 적합합니다.',
'A라인, 프린세스',
'슬림 (상체가 더 넓어 보일 수 있음)');

-- 4. 마른 체형
INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses) VALUES
('마른 체형',
'슬림한 체형에는 프린세스 라인이 잘 어울립니다.',
'풍성한 스커트가 체형을 보완해주고, 전체적으로 사랑스러운 이미지를 만들어줍니다.',
'프린세스',
'슬림 (마른 체형이 더 마르게 보일 수 있음)');

-- 5. 팔 라인이 신경 쓰이는 체형
INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses) VALUES
('팔 라인이 신경 쓰이는 체형',
'팔 라인이 고민된다면 상체를 너무 타이트하게 드러내는 슬림 라인보다는, 상체를 적당히 감싸주고 스커트가 퍼지는 A라인이나 벨라인이 안정감 있게 연출해 줍니다.',
'상체를 적당히 감싸주는 디자인이 적합합니다.',
'A라인, 벨라인',
'슬림 (팔 라인이 노출될 수 있음)');

-- 6. 허리가 짧은 체형
INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses) VALUES
('허리가 짧은 체형',
'허리선이 살짝 내려간 드롭 웨이스트 드레스는 하체를 길어 보이게 만들어 전체적인 비율을 맞춰줍니다.',
'허리선을 낮춰서 하체를 길어 보이게 하는 스타일이 적합합니다.',
'드롭 웨이스트',
'하이웨이스트 (허리가 더 짧아 보일 수 있음)');

-- 7. 복부가 신경 쓰이는 체형
INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses) VALUES
('복부가 신경 쓰이는 체형',
'복부를 자연스럽게 커버하려면 A라인 드레스가 최적입니다.',
'허리에서 자연스럽게 퍼지는 라인이 체형 커버에 탁월합니다.',
'A라인',
'슬림 (복부가 노출될 수 있음), 머메이드 (복부 라인이 강조될 수 있음)');

-- 8. 키가 큰 체형
INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses) VALUES
('키가 큰 체형',
'키가 큰 신부님은 긴 기장의 슬림 드레스가 우아함을 더해줍니다.',
'특히 미니멀한 디자인은 세련된 이미지를 강조해줍니다.',
'슬림',
'엠파이어 라인 (키가 더 커 보일 수 있음)');

-- 9. 어깨가 좁은 체형
INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses) VALUES
('어깨가 좁은 체형',
'어깨가 좁다면 상체에 볼륨이 살아나는 프린세스 라인이나 벨라인이 균형감을 잡아줍니다.',
'상체에 레이스나 셔링 같은 디테일이 들어간 A라인 드레스도 어깨와 상체 라인을 보완해 주는 데 도움이 됩니다.',
'프린세스, 벨라인, A라인',
'슬림 (어깨가 더 좁아 보일 수 있음)');

-- 10. 체형 전체를 커버하고 싶은 경우
INSERT INTO body_type_definitions (body_feature, strengths, style_tips, recommended_dresses, avoid_dresses) VALUES
('체형 전체를 커버하고 싶은 경우',
'체형 고민이 많을 때는 클래식한 벨라인 드레스가 가장 안전한 선택입니다.',
'로맨틱하면서도 웅장한 분위기를 연출할 수 있습니다.',
'벨라인',
'특별히 피해야 할 스타일은 없으나, 체형의 특성에 따라 선택적으로 피하는 것이 좋습니다.');
