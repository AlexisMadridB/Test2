"""Evaluación offline del baseline de TutorIA (sin llamadas a ningún LLM).

Ragas devolvió NaN en la corrida baseline (el LLM juez falló), pero el pipeline sí guardó, por pregunta,
la respuesta del RAG y los 5 fragmentos recuperados (eval/results/baseline_detalle.csv).
Con esos datos reales se calculan las 4 métricas con las MISMAS definiciones de Ragas, pero con
anotaciones explícitas en lugar de un juez LLM:

  context_recall    = afirmaciones de la respuesta de referencia que aparecen en los fragmentos / total
  context_precision = precisión promedio (AP@k) con la relevancia de cada fragmento según esas mismas afirmaciones
  faithfulness      = afirmaciones de la respuesta que se sostienen en los fragmentos / total (revisión manual)
  answer_relevancy  = respuestas que contestan la pregunta planteada / total (revisión manual)

Uso:  python evaluacion_manual.py eval/results/baseline_detalle.csv
"""
import ast, re, sys, unicodedata, json
import pandas as pd

def norm(t):
    t = unicodedata.normalize("NFKD", str(t)).lower()
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t)

# Afirmaciones de la referencia (regex sobre texto normalizado) por pregunta
REF = {
 "Q01": [r"create database", r"if not exists"],
 "Q02": [r"comando use", r"no (la )?selecciona|automaticamente", r"crear tablas, insertar"],
 "Q03": [r"drop database", r"permanente|irreversible|destructiv", r"tablas y registros"],
 "Q04": [r"todas las condiciones|ambas condiciones|and:", r"al menos una"],
 "Q05": [r"cero, uno o|cero o mas", r"un solo caracter|unico caracter|exactamente un"],
 "Q06": [r"is null", r"signo igual|= null", r"campo vacio"],
 "Q07": [r"orden arbitrario", r"siempre (acompanalo|ordenar)"],
 "Q08": [r"limit 10 offset 10", r"order by"],
 "Q09": [r"delimiter //", r"terminar una instruccion", r"delimiter ;"],
 "Q10": [r"in:.{0,20}por defecto", r"out:", r"inout"],
 "Q11": [r"con call|se llaman con call", r"un solo valor", r"modificacion \(dml\)"],
 "Q12": [r"se ejecuta automaticamente", r"before|after", r"old"],
 "Q13": [r"fila \(registro\)", r"columna \(campo\)"],
 "Q14": [r"\bchar\b.{0,120}fij", r"varchar.{0,160}variable|variable.{0,60}varchar"],
 "Q15": [r"sin escribir la clausula where", r"todas las filas"],
 "Q16": [r"llave primaria", r"identific|unic", r"nula"],
 "Q17": [r"inner join", r"left join", r"valores null"],
}

# Afirmaciones de la respuesta: (¿se sostiene en el contexto?, regex de comprobación, descripción)
# Etiqueta manual (S=sostenida / N=no está en los fragmentos). La regex solo es una verificación cruzada.
FAITH = {
 "Q01": [("S","create database","CREATE DATABASE crea la BD"),("S","if not exists","IF NOT EXISTS evita el error"),
         ("S","solo si no existe","se crea solo si no existe"),("S","contenedor","la BD es un contenedor"),
         ("N","idempot|varias veces","útil para scripts idempotentes")],
 "Q02": [("S","comando use","USE selecciona la BD activa"),("S","crear tablas, insertar","las operaciones siguientes usan esa BD"),
         ("S","automaticamente|selecciona","no se elige BD automáticamente"),("N","debe existir previamente","la BD debe existir antes"),
         ("N","devuelve un error|error.{0,30}no existe","error si no existe"),("N","sin cerrar la conexion|cambiar de contexto","cambiar de proyecto sin cerrar conexión"),
         ("N","fallaria","sin USE, CREATE TABLE fallaría")],
 "Q03": [("S","drop database","DROP DATABASE elimina la BD"),("S","tablas y registros","con sus tablas y registros"),("S","destructiva y eliminara permanentemente","operación destructiva y permanente"),
         ("N","directorio","borra el directorio y archivos"),("N","liberan|libera","libera recursos"),("N","privilegio","requiere privilegio DROP"),
         ("N","cerrara|conexiones","cierra conexiones activas"),("N","if exists","DROP DATABASE IF EXISTS"),
         ("N","mysqldump|respaldo","backups con mysqldump"),("N","desaparece|ya no aparece","desaparece de SHOW DATABASES"),
         ("N","1049|no encontrada","USE posterior da error 1049")],
 "Q04": [("S","and","AND: todas las condiciones"),("S","al menos","OR: al menos una"),("S","categoria = 'electronica'","ejemplo AND del material"),
         ("S","espana","ejemplo OR del material"),("S","parentesis","usar paréntesis para agrupar (está en el material)")],
 "Q05": [("S","cero, uno o|cero o","% = cero, uno o varios"),("S","un solo caracter|exactamente un|unico","_ = un carácter"),
         ("S","like 'a%'","ejemplo LIKE 'A%'"),("S","%gmail%","ejemplo '%gmail%'"),("N","like 'a_'","ejemplo LIKE 'A_' con 'Ab','A1'")],
 "Q06": [("S","ausente|desconocido","NULL = dato ausente o desconocido"),("S","signo igual|= null","no se usa ="),
         ("S","is not null","IS NOT NULL"),("S","is null","IS NULL")],
 "Q07": [("S","limit","LIMIT restringe filas"),("S","orden arbitrario","sin ORDER BY el orden es arbitrario"),
         ("N","union|indices","el orden interno cambia por índices/UNION"),("S","paginacion","paginación fiable"),
         ("S","siempre ordenar antes de limitar","cita de buena práctica")],
 "Q08": [("S","limit y offset|offset","OFFSET salta filas"),("S","order by","ordenar antes"),
         ("S","limit 10 offset 10","OFFSET 10 LIMIT 10"),("N","n - 1|(n-1)","fórmula OFFSET=(n-1)*p"),("S","usuarios order by id","ejemplo del material")],
 "Q09": [("S","delimiter //","DELIMITER // cambia el delimitador"),("S","terminar una instruccion","; termina la instrucción"),
         ("S","prematuramente","si no, termina prematuramente"),("S","delimiter ;|restauramos","se restaura con DELIMITER ;"),("S","consultarsaldo","ejemplo con saldo")],
 "Q10": [("S","in:","IN por defecto"),("S","out:","OUT devuelve valor"),("S","inout","INOUT entra y sale"),
         ("N","no puede cambiar el valor","IN no altera la variable externa")],
 "Q11": [("S","call","procedimiento con CALL"),("S","un solo valor","función devuelve un valor"),("S","dentro de otras sentencias","función dentro de SELECT"),
         ("S","parametros out","procedimiento puede no devolver o usar OUT"),("S","insert, update","procedimiento permite DML"),
         ("S","prohibida","función no puede modificar datos"),("S","deterministic","DETERMINISTIC / NOT DETERMINISTIC"),
         ("S","multiples conjuntos","puede devolver varios conjuntos"),("N","aplicaciones o scripts","uso típico desde aplicaciones")],
 "Q12": [("S","se ejecuta automaticamente","trigger automático"),("S","insert","INSERT/UPDATE/DELETE"),("S","before|after","BEFORE/AFTER"),
         ("S","new:","NEW"),("S","old","OLD"),("S",r"for\s+each row","FOR EACH ROW"),("S","auditar_cambio_salario","ejemplo del material")],
 "Q13": [("S",r"fila \(registro\)","fila = registro"),("S",r"columna \(campo\)","columna = campo/atributo"),
         ("S","columnas y","tabla con filas y columnas"),("S","llave foranea","FK referencia a la PK"),("S","tipos de datos","columnas con tipo de dato")],
 "Q14": [("S","longitud variable","VARCHAR es de longitud variable"),("S","ideal para nombres","VARCHAR ideal para nombres/correos"),
         ("N","rellen|espacios en blanco","CHAR rellena con espacios"),("N","1 o 2 bytes","VARCHAR usa 1 o 2 bytes de longitud"),
         ("N","rendimiento","CHAR es más rápido"),("N","ahorra espacio","VARCHAR ahorra espacio")],
 "Q15": [("S","clausula where","WHERE filtra filas"),("S","todas las filas","sin WHERE afecta todas"),("S","delete from clientes;","DELETE sin WHERE vacía la tabla"),
         ("S","update","UPDATE sin WHERE cambia todo"),("S","por error","riesgo de error grave"),("N","rendimiento","impacto en rendimiento")],
 "Q16": [("S","llave primaria","PK identifica de forma única"),("S","nula","no puede ser nula"),("N","btree","índice BTREE por defecto"),
         ("N","indice unico","crea un índice único automático"),("S","llave foranea","FK referencia a la PK"),("S","auto_increment","AUTO_INCREMENT")],
 "Q17": [("S","inner join","INNER JOIN solo coincidencias"),("S","left join","LEFT JOIN todas las de la izquierda"),
         ("S","valores null","NULL en el lado derecho"),("S","pedidos","ejemplo clientes/pedidos"),("S","inner join","cliente sin pedidos no sale en INNER")],
}
# Revisión manual: ¿la respuesta contesta lo que se preguntó y coincide con la referencia? (las 17 sí)
RELEVANT = {q: True for q in REF}

def ap_precision(flags):
    hits, s = 0, 0.0
    for k, r in enumerate(flags, 1):
        if r:
            hits += 1
            s += hits / k
    return s / hits if hits else 0.0

def main(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    rows, warns = [], []
    for _, r in df.iterrows():
        q = r["id"]
        ctxs = [norm(c) for c in ast.literal_eval(r["retrieved_contexts"])]
        alltxt = " ".join(ctxs)
        # recall
        found = [bool(re.search(p, alltxt)) for p in REF[q]]
        recall = sum(found) / len(found)
        # precision
        rel = [any(re.search(p, c) for p in REF[q]) for c in ctxs]
        prec = ap_precision(rel)
        # faithfulness
        sup = 0
        for lab, rx, desc in FAITH[q]:
            present = bool(re.search(rx, alltxt))
            if lab == "S": sup += 1
            if (lab == "S") != present:
                warns.append(f"{q}: '{desc}' etiqueta={lab} pero regex {'encontrada' if present else 'NO encontrada'}")
        faith = sup / len(FAITH[q])
        rows.append(dict(id=q, faithfulness=round(faith,3), answer_relevancy=1.0 if RELEVANT[q] else 0.0,
                         context_precision=round(prec,3), context_recall=round(recall,3),
                         afirm_ref_ok=f"{sum(found)}/{len(found)}", afirm_resp_ok=f"{sup}/{len(FAITH[q])}",
                         frag_relevantes=f"{sum(rel)}/5"))
    out = pd.DataFrame(rows)
    return out, warns

if __name__ == "__main__":
    out, warns = main(sys.argv[1])
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print("\nPROMEDIOS:"); print(out[["faithfulness","answer_relevancy","context_precision","context_recall"]].mean().round(3).to_string())
    print("\nAVISOS (etiqueta manual vs regex):"); print("\n".join(warns) or "ninguno")
    out.to_csv(sys.argv[2] if len(sys.argv)>2 else "evaluacion_manual_baseline.csv", index=False, encoding="utf-8-sig")
