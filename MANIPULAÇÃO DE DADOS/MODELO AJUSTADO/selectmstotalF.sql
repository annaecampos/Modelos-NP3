SELECT
    CAST(
        (SUM(p.mstotal * (ST_Area(s.subpoligono) / ST_Area(t.poligono)))
         / SUM(ST_Area(s.subpoligono) / ST_Area(t.poligono)))
        AS NUMERIC(7,2)
    ) AS mstotal
FROM subarea s
JOIN potreiro t ON s.idpotreiro = t.id
JOIN pastagem p ON p.idsubarea  = s.id
JOIN medicao  m ON p.idmedicao  = m.id
WHERE m.data       IS NOT NULL
  AND p.mstotal    != 0
  AND p.dentrofora  = 'F'
  AND t.id = 1
GROUP BY p.dentrofora, m.data
ORDER BY m.data
