SELECT
    p.dentrofora,
    m.data,
    CAST(
        (SUM(p.media * (ST_Area(s.subpoligono) / ST_Area(t.poligono)))
         / SUM(ST_Area(s.subpoligono) / ST_Area(t.poligono)))
        AS NUMERIC(7,2)
    ) AS media,
    CAST(
        (SUM(p.mstotal * (ST_Area(s.subpoligono) / ST_Area(t.poligono)))
         / SUM(ST_Area(s.subpoligono) / ST_Area(t.poligono)))
        AS NUMERIC(7,2)
    ) AS mstotal,
    CAST(
        (SUM((p.msanoni / (p.msanoni + p.msoutras))
             * (ST_Area(s.subpoligono) / ST_Area(t.poligono)))
         / SUM(ST_Area(s.subpoligono) / ST_Area(t.poligono)))
        AS NUMERIC(7,2)
    ) AS pcanonni,
    EXTRACT(MONTH FROM m.data) AS mes,
    EXTRACT(YEAR  FROM m.data) AS ano,
    p.idpotreiro
FROM subarea s
JOIN potreiro t ON s.idpotreiro = t.id
JOIN pastagem p ON p.idsubarea  = s.id
JOIN medicao  m ON p.idmedicao  = m.id
WHERE m.data    IS NOT NULL
  AND p.mstotal != 0
  AND t.id = 1          -- 1=p20inf | 2=p20mira | 3=p21inf | 4=p21mira
GROUP BY p.dentrofora, m.data, p.idpotreiro
ORDER BY m.data
