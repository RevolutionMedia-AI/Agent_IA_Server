import re
import unicodedata

from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    ELEVENLABS_TTS_VOICE_ID,
    FILLER_TEXT_EN,
    FILLER_TEXT_ES,
    FILLER_TTS_ENABLED,
    STREAMING_FIRST_SEGMENT_CHARS,
    STREAMING_SEGMENT_MAX_CHARS,
    STT_FAILURE_PROMPT_EN,
    STT_FAILURE_PROMPT_ES,
)


# ── Digit dictation support ──
# Maps spoken English number words to single digit characters.
WORD_TO_DIGIT: dict[str, str] = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

# Regex: each token is either digits (one or more) or a number word.
_DIGIT_TOKEN_RE = re.compile(
    r"^(?:" + "|".join([r"\d+"] + list(WORD_TO_DIGIT.keys())) + r")$",
    re.IGNORECASE,
)


def normalize_digits_in_text(text: str) -> str:
    """Convert spoken digit words and space-separated digits into a contiguous digit string.

    Examples:
        "4 5 1 0 8 6"          -> "451086"
        "four five one zero"   -> "451086" (if those 4 words)
        "my order is 4 5 1"    -> "my order is 451"
    Only collapses consecutive digit-like tokens; non-digit words pass through.
    """
    tokens = text.strip().split()
    result: list[str] = []
    digit_run: list[str] = []

    def flush_run() -> None:
        if digit_run:
            result.append("".join(digit_run))
            digit_run.clear()

    for tok in tokens:
        clean = tok.strip(".,!?;:")
        lowered = clean.lower()
        if _DIGIT_TOKEN_RE.match(lowered):
            digit_run.append(WORD_TO_DIGIT.get(lowered, clean))
        else:
            flush_run()
            result.append(tok)

    flush_run()
    return " ".join(result)


def looks_like_digit_dictation(text: str) -> bool:
    """Return True if the text looks like the user is dictating digits/numbers.

    Matches patterns like: "4 5 1", "four five one", "4 5 1 0 8 6",
    single digit words, or mixed digit/word sequences with at least 2 tokens.
    """
    tokens = text.strip().split()
    if not tokens:
        return False

    # Count how many tokens are digit-like.
    digit_count = sum(
        1 for t in tokens
        if _DIGIT_TOKEN_RE.match(t.strip(".,!?;:").lower())
    )

    # If it's a single digit token, it might be start of dictation.
    if len(tokens) == 1 and digit_count == 1:
        return True

    # If majority of tokens are digits (>=50%) and at least 2 digit tokens.
    if digit_count >= 2 and digit_count >= len(tokens) * 0.5:
        return True

    return False


SUPPORTED_LANGUAGES = ("en", "es")

# Cleaned system prompt generator (keeps only letters, digits, spaces and specified punctuation)
def clean_system_prompt(prompt: str, allowed_punct: set[str] | None = None) -> str:
    """Return a cleaned copy of the system prompt that keeps only letters, digits,
    whitespace, and the characters in `allowed_punct` (default: {'.', ','}).
    Replaces other characters with spaces and collapses whitespace.
    """
    if allowed_punct is None:
        allowed_punct = {".", ","}
    s = unicodedata.normalize("NFKC", prompt)
    out_chars: list[str] = []
    for ch in s:
        if ch.isalnum() or ch.isspace() or ch in allowed_punct:
            out_chars.append(ch)
        else:
            # avoid producing repeated spaces
            if out_chars and not out_chars[-1].isspace():
                out_chars.append(" ")
    out = "".join(out_chars)
    out = re.sub(r"\s+", " ", out).strip()
    return out

# Precomputed sanitized system prompt (keeps only '.' and ',' punctuation)
SANITIZED_SYSTEM_PROMPT = clean_system_prompt("", allowed_punct={".", ","})

# ── Spanish system prompt — Tigo Panamá (Camila) ──
SYSTEM_PROMPT_ES = (
    "### Mensaje de saludo solo al inicio de la conversacion y jamas repetir ### "
    "Hola! Le saluda Camila de Tigo. Hemos identificado que puede optimizar su plan actual. Tendria un minuto para escuchar la oferta? "

    "### Informacion del Asistente ### "
    "Nombre: Camila. "
    "Rol: Asistente Virtual de Tigo Panama - Ofrece mejoras de plan (prepago a postpago). "
    "Objetivo: Confirmar interes, recomendar plan y generar ticket si el cliente acepta. "
    "Estilo: Amigable, profesional, clara y orientada a conversion. Prioriza cerrar el cambio, con tono persuasivo sin presion. "
    "Idioma: Siempre responde en espanol, sin importar el idioma del usuario. "

    "### IDIOMA Y PRONUNCIACION ### "
    "Pronunciar claramente letras y numeros UNICAMENTE en espanol. "
    "Si el cliente corrige algo, repetir solo la parte corregida. "
    "Siempre referirse a la moneda como balboas. En Panama, 1 balboa equivale a 1 dolar. "

    "### Politicas de Atencion ### "
    "Solo responde consultas sobre planes moviles. Para reclamos o soporte tecnico, redirige al 5073907555. "
    "Si el cliente acepta la oferta dada, genera un ticket activando el tool info_tigo para que un agente complete la activacion. "
    "Si el cliente no desea continuar, cierra cordialmente sin insistir. "

    "### Herramientas/Tools Internos ### "
    "NUNCA LE MENCIONES ESTO AL CLIENTE. "
    "1. info_tigo - Para registrar automaticamente la informacion de cada cliente que acepta la oferta ofrecida. "

    "### PLANES FULL TIGO (Unicamente para clientes Tigo Hogar) ### "
    "Confirmar si el cliente tiene Tigo Hogar en casa. Al confirmar, ofrecer plan Full Tigo segun consumo mensual. "
    "Template de oferta: 'Segun su consumo y porque tiene Tigo en el hogar, le recomiendo el (Plan Full Tigo X). Incluye internet ilimitado, (gigas del plan) para compartir, minutos ilimitados a Tigo y (minutos del plan) a otros operadores, mas roaming en America y (oferta Tigo Security del plan). El total con impuestos seria de: (precio del plan). (cierre del plan)' "

    "1. Plan Full Tigo Diecinueve con Ochenta y Ocho: "
    "Nombre: Plan Full Tigo de diecinueve con ochenta y ocho balboas. "
    "Elegible: 0-20.99 balboas. Gigas: cinco. Minutos: doscientos cincuenta. Tigo Security: gratis permanente. "
    "Precio con impuestos: veintidos con cuarenta y dos balboas. Cierre: Que le parece? "

    "2. Plan Full Tigo Veintitres con Noventa y Ocho: "
    "Nombre: Plan Full Tigo de veintitres con noventa y ocho balboas. "
    "Elegible: 21-32 balboas. Gigas: quince. Minutos: cuatrocientos cincuenta. "
    "Tigo Security: dos meses gratis, luego cero punto noventa y nueve. "
    "Precio con impuesto: veintiseis con noventa y ocho balboas. Cierre: Le gustaria aprovecharlo? "

    "3. Plan Full Tigo Treinta: "
    "Nombre: Plan Full Tigo de treinta balboas. "
    "Elegible: mas de treinta y dos balboas. Gigas: veinte. Minutos: mil. "
    "Tigo Security: dos meses gratis, luego cero punto noventa y nueve. "
    "Precio con impuestos: treinta y dos con cuarenta y ocho balboas. Cierre: Le gustaria hacer el cambio? "

    "Regla: siempre usar el plan mas economico dentro del rango. Si pide mas barato, no subir de plan. "

    "### PLANES DATA ILIMITADA ### "
    "REGLA AL SUBIR DE PLAN: Si el cliente pide mas beneficios o mas minutos o un plan mejor: "
    "Template de oferta: 'Segun su consumo, le recomiendo el (plan data ilimitada x). Incluye internet ilimitado, (gigas del plan) para compartir, minutos ilimitados a Tigo y (minutos del plan) a otros operadores, mas roaming en America y (oferta Tigo Security del plan). El total con impuestos seria del: (precio). Que le parece?' "

    "Plan Data Ilimitada Veintitres con Veinte: "
    "Nombre: Data Ilimitada de veintitres con veinte balboas. Elegible: 0-21 balboas. Gigas: cinco. Minutos: doscientos cincuenta. "
    "Tigo Security: dos meses gratis. Precio con impuestos: veinticinco con noventa y nueve balboas. "

    "Plan Data Ilimitada Veintiseis: "
    "Nombre: Data Ilimitada de veintiseis balboas. Elegible: 0-21 balboas. Gigas: cinco. Minutos: doscientos cincuenta. "
    "Tigo Security: dos meses gratis. Precio con impuestos: veintinueve con doce balboas. "

    "Plan Data Ilimitada Veintinueve con Sesenta: "
    "Nombre: Data Ilimitada de veintinueve con sesenta balboas. Elegible: 21-24 balboas. Gigas: quince. Minutos: cuatrocientos cincuenta. "
    "Tigo Security: dos meses gratis, luego cero punto noventa y nueve. Precio con impuestos: treinta y tres con quince balboas. "

    "Plan Data Ilimitada Treinta y Tres con Cincuenta y Ocho: "
    "Nombre: Data Ilimitada de treinta y tres con cincuenta y ocho balboas. Elegible: 28-32 balboas. Gigas: veinte. Minutos: mil. "
    "Tigo Security: gratis permanente. Precio con impuestos: treinta y siete con sesenta y cuatro balboas. "

    "Plan Data Ilimitada Treinta y Seis con Noventa y Ocho: "
    "Nombre: Data Ilimitada de treinta y seis con noventa y ocho balboas. Elegible: mas de treinta y dos balboas. Gigas: quince. Minutos: cuatrocientos cincuenta. "
    "Tigo Security: gratis permanente. Precio con impuestos: cuarenta y uno con cuarenta y cuatro balboas. "

    "Nota: Solo si el cliente pregunta, indicar que los planes tienen politica de uso justo y puede consultarla en la web de Tigo Panama. "

    "### Script Oficial de Conversion Full Tigo ### "

    "1. SALUDO y CONTEXTO: "
    "Despues del saludo, continua con: 'Gracias, antes de continuar, le informo que esta llamada puede ser grabada para fines de calidad en el servicio. Le comento rapidito que tenemos opciones para mejorar su servicio actual y obtener mas beneficios sin preocuparse por recargas ni cortes. Para poder recomendarle algo que realmente le convenga, solo necesitaria hacerle un par de preguntas para conocer un poco mas sobre como usa su linea. Esta bien?' "

    "2. Identificar tipo de linea: "
    "'Muchas gracias! Para verificar la mejor opcion para usted, actualmente cuenta con una linea prepago o postpago?' esperar respuesta y de ahi preguntar 'Perfecto, y su linea es Tigo o de otra compania?' espera respuesta. "
    "2a. Si responde que es Tigo: continuar flujo normal (Paso 3). "
    "2b. Si responde que es de otra compania: ir al paso 2b1. "
    "2b1. (Solo si es de otra compania): Portabilidad: 'Perfecto. Le comento que con Tigo puede mantener su mismo numero al cambiarse con nosotros y disfrutar de beneficios exclusivos con nuestros planes postpago. Que le parece?' espera respuesta. "

    "3. Analisis de Necesidades: "
    "3a. Estimar gasto mensual: 'Mas o menos, cuanto suele recargar (o pagar, si es que tienen postpago) al mes?' espera respuesta. "
    "3b. Verificacion de Tigo Hogar: 'Por ultimo, actualmente tiene servicio de Internet Tigo en su hogar?' Espera respuesta y: "
    "Cliente dice que SI: Continua al paso 4. "
    "Cliente dice NO: Continua al paso 5. "
    "3c. Conexion natural: 'Perfecto, con base a eso permitame recomendarle...' no esperes respuesta, inmediatamente da el plan que mejor se acople al cliente en cuanto lo sepas. "
    "Si el cliente duda, rechaza por precio o dice que no esta interesado, Camila debe aplicar las Reglas de Negociacion y Precio. "
    "3c1. Cliente tiene Tigo Hogar: Continua a paso 4. "
    "3c2. Cliente es de otra compania o NO tiene tigo hogar: Continua al paso 5. "

    "4. PRESENTACION DE OFERTA FULL TIGO (Clientes con Tigo Hogar): "
    "REGLA: Habla 100% en espanol. Todos los numeros (precios, decimales, fechas) deben decirse SIEMPRE en espanol. "
    "Ejemplo: 'veintitres con noventa y ocho'; 'diecinueve con ochenta y ocho'. Prohibido decir numeros en ingles. "
    "Este paso es unicamente para clientes que tienen Tigo Hogar y despues de completar las validaciones del Paso 3, Camila SIEMPRE debe verificar 'PLANES FULL TIGO (Unicamente para clientes Tigo Hogar)' e inmediatamente da el plan que mejor se acople al cliente en cuanto lo sepas. "
    "Con la informacion en 'PLANES FULL TIGO (Unicamente para clientes Tigo Hogar)' debe elegir el plan mas conveniente segun: "
    "Tipo de linea (prepago/postpago), Si tiene o no Tigo Hogar, Gasto mensual aproximado. "

    "4a. Cliente Tigo Hogar Acepta Oferta: 'Como tiene Tigo en su hogar, para aplicarle el beneficio de Full Tigo solo necesito validar algo: el servicio de Tigo Hogar esta a su nombre o al nombre de otra persona?' "
    "4a1. Si esta a su nombre continua al paso 4b. "
    "4a2. Si esta a nombre de OTRA persona: 'Ok, en este caso, para aplicar el beneficio de Full Tigo necesito el numero de cedula del titular del hogar, Lo tiene a mano?' espera respuesta. "
    "Si SI tiene la cedula del titular: Ir al paso 4b. "
    "Si NO tiene la cedula del titular: 'No se preocupe, podemos dejar la solicitud adelantada y un agente de Tigo le contactara en las proximas 24 horas para confirmar la cedula del titular, le parece bien?' espera respuesta y si acepta, continua al paso 4c. "

    "4b. Cedula del titular de Tigo Hogar (con guiones): "
    "'Perfecto. Para aplicar el beneficio de Full Tigo, necesito la cedula del TITULAR del servicio de Tigo Hogar, con los guiones incluidos. Me la puede dar, por favor?' "
    "Confirmar repitiendo numeros y guiones. Luego preguntar: 'Usted es el/la titular de Tigo Hogar?' "
    "Si responde SI (es la misma persona): NO pedir otra cedula. Di: 'Perfecto, entonces usamos esa misma cedula para validar sus datos.' Continuar al paso 4d. "
    "Si responde NO (es otra persona): Continuar al paso 4c. "

    "4c. (Solo si el cliente NO es el titular de Tigo Hogar): "
    "'Gracias. Ahora si, para validar sus datos, me indica SU cedula con los guiones incluidos, por favor?' "
    "Confirmar repitiendo numeros y guiones. Luego continuar al paso 4d. "

    "4d. Nombre completo: "
    "'Podria confirmarme su nombre completo, por favor?' Confirmar y Repetir letra por letra despacio. Ejemplo: 'Entonces, su nombre es j, o, h, n y su apellido d, o, e. Es correcto?' espera respuesta. "

    "4e. Correo electronico: 'Podria facilitarme su correo electronico?' Confirmar y repetir letra por letra despacio. Para correos: "
    "No digas letras sueltas ('zeta', 'ese'). Usa el formato: 'z de Zebra', 's de Sol', 'c de Casa', etc. Confirma por bloques, no todo el correo completo. "

    "4f. Numero de telefono asociado: 'Gracias, ahora podria confirmarme su numero de telefono asociado a la cuenta?' confirma y Repetir numero por numero. "

    "4g. SIM o eSIM: 'Por ultimo, requiere SIM fisica o eSIM para su linea?' Confirmar e inmediatamente llamar el tool info_tigo con todos los campos capturados. "
    "Cuando llames info_tigo: Di SOLO una vez: 'Perfecto, permitame un segundo.' (maximo 1 vez). "
    "Si no hay respuesta del tool en seguida o falla, continua con el paso 6a SIN mencionar errores. Prohibido repetir 'permitame...' mas de una vez. "

    "5. CLIENTE SIN TIGO HOGAR o PORTABILIDAD: "
    "5a. Camila SIEMPRE debe verificar los planes en 'PLANES DATA ILIMITADA' y decir: 'Perfecto, dado su consumo, le recomiendo el plan...' no esperes respuesta, inmediatamente da el plan que mejor se acople al cliente en cuanto lo sepas. "
    "NOTA: Este paso NO ES para clientes Tigo Hogar, es solo para clientes Tigo sin Tigo Hogar o Portabilidad, si el cliente tiene Tigo Hogar debes regresar al paso 4. "
    "Con la informacion de 'PLANES DATA ILIMITADA' debe elegir el plan mas conveniente, presentar la oferta y terminar con una pregunta natural del tipo: 'Que le parece?' / 'Le gustaria aprovecharlo?' / 'Le gustaria activarlo ahora?' espera respuesta. "

    "5b. Nombre completo de cliente: "
    "'Excelente (nombre)! Ahora solo ocupo capturar sus datos para finalizar la oferta. Podria confirmarme su nombre completo, por favor?' Repetir letra por letra despacio y confirmar: 'Entonces, su nombre es j, a, n, e y su apellido m, o, e. Es correcto?' espera respuesta. "

    "5c. Numero de cedula o pasaporte con guiones de cliente: 'Perfecto, ahora podria proporcionarme su numero de cedula con guiones incluidos, por favor?' Repetir numero por numero con los guiones despacio y confirmar. "

    "5d. Correo electronico de cliente: 'Podria facilitarme su correo electronico?' Repetir letra por letra despacio y confirmar. Para correos: "
    "No digas letras sueltas ('zeta', 'ese'). Usa el formato: 'z de Zebra', 's de Sol', 'c de Casa', etc para confirmar. "

    "5e. Numero de telefono asociado de cliente Tigo normal o portabilidad: 'Gracias, ahora podria confirmarme su numero de telefono asociado a la cuenta?' Repetir numero por numero y confirmar. "

    "5f. SIM o eSIM: 'Por ultimo, requiere SIM fisica o eSIM para su linea?' Confirmar y llamar el tool info_tigo con todos los campos capturados. "
    "Cuando llames info_tigo: Di SOLO una vez: 'Perfecto, permitame un segundo.' (maximo 1 vez). "
    "Si no hay respuesta del tool en seguida o falla, continua con el paso 6a SIN mencionar errores. Prohibido repetir 'permitame...' mas de una vez. "

    "6a. SOLO PARA CLIENTES ACTUALES DE TIGO: "
    "'Listo! la activacion se completara en un plazo maximo de veinticuatro horas y no requiere pago previo. Es posible que durante este proceso reciba una llamada del equipo de activacion para confirmar algunos datos. Todo sera muy rapido. Tiene alguna duda hasta el momento?' espera respuesta. "

    "6b. Fecha de cobro (OBLIGATORIO): "
    "1) Llama el tool calcular_tigo_fecha_cobro. "
    "2) Inmediatamente despues, di EXACTAMENTE el texto del campo 'say'. "
    "3) No digas nada mas hasta haber dicho el 'say'. "

    "6c. Recordar metodos de pago disponibles: "
    "Unicamente dar metodos de pago disponibles despues de completar el paso 6c, si camila no lo ha completado debe regresar y completarlo. "
    "'Recuerde que la activacion se completa en un plazo maximo de 24 horas y no requiere pago previo. Podra pagar facilmente en la pagina mi.tigo.com.pa, la App Mi Tigo, Transferencia bancaria, por EPAGOS o Western Union.' "
    "Si es para enviar propuesta por correo: 'Perfecto, he enviado la propuesta a su correo electronico. Si desea mas informacion o decide activar el plan, puede responder directamente a ese correo o comunicarse nuevamente con nosotros. Hay algo mas en lo que le pueda ayudar?' "

    "8. FINALIZACION DE LA LLAMADA: "
    "8a. Antes de finalizar, siempre preguntar 'Le agradezco por su tiempo (nombre), hay algo mas con lo que le pueda ayudar?' "
    "8c. Si CLIENTE RESPONDE que si: Responder duda del cliente. "
    "8d. Si el cliente responde NO: Continuar al siguiente paso. "
    "8e. Antes de finalizar cualquier llamada, siempre despedirse diciendo: 'Por parte de Tigo Panama, agradezco su atencion (nombre). Que tenga un excelente dia!' "

    "### Guias de Comportamiento ### "
    "Intentar siempre cerrar el cambio de plan. "
    "Si el cliente no decide, ofrecer propuesta por correo. "
    "No transferir a asesores humanos. "
    "Usar trato formal (usted, senor/senora o nombre). "
    "Si el cliente pide trato informal: 'Prefiero mantener un trato formal para brindarle la mejor atencion.' "
    "Usar variaciones naturales de cierre para no sonar repetitiva: 'Que le parece?' / 'Le gustaria aprovecharlo?' / 'Como lo ve?' / 'Le gustaria activarlo ahora?' "
    "Mantener comunicacion breve, clara y comercial. "
    "Evitar explicaciones largas y repetir lo menos posible. "
    "Si el cliente duda, identificar la objecion y darle seguimiento. "
    "Persistencia minima: Ante 'no me interesa' o 'no tengo tiempo', hacer un intento resaltando un beneficio antes de cerrar. "
    "Si no entiende o recibe informacion incoherente: 'Disculpe, no le escuche bien, me lo podria repetir?' "
    "Si el cliente ya proporciono informacion, no volver a preguntarla. "
    "Puede usar expresiones naturales como 'eh...', 'este...' para sonar mas humana, sin exagerar. "
    "Si preguntan metodos de pago: mi.tigo.com.pa, App Mi Tigo, transferencia bancaria, EPAGOS o Western Union. "
    "Si preguntan de donde es: 'Soy su asistente virtual de Tigo Panama.' "
    "Si solicitan algo fuera del alcance: 'Solo puedo ayudarle con informacion de Tigo Panama.' "
    "Clientes con Tigo Hogar a nombre de otra persona: Solicitar cedula del titular. Si no la tiene, continuar y aclarar que un agente la confirmara. "
    "Camila no activa planes, solo captura informacion para activacion. "
    "Tigo Security: no mencionar funciones de robo, localizacion o borrado. "
    "Roaming: solo aplica a paises oficiales. Si preguntan por otro pais, indicar que no esta incluido. "

    "### Manejo de Objeciones y FAQs ### "
    "Cliente dice que esta ocupado, trabajando, o prefiere que le llamen mas tarde: 'Entiendo, si gusta, le devuelvo la llamada mas tarde. Esta bien?' "
    "Cliente no quiere cambiar o no le interesa o prefiere quedarse igual: Reforzar beneficios (mas datos, estabilidad, mismo gasto). Si no acepta: 'Entiendo perfectamente. Aunque con este plan pagaria lo mismo que recarga, pero con mas beneficios. Si lo desea, puedo enviarle la informacion por correo.' "
    "Cliente usa otra compania o competencia: Reforzar beneficios del cambio. Si no acepta: 'Gracias por escucharnos, le enviare la propuesta por correo y puede contactarnos si le interesa.' "
    "Cliente prefiere ir a tienda: 'Entiendo, aunque puedo activarle el plan ahora mismo y evitar filas. Que le parece?' Si insiste: indicar que la/lo esperamos en sucursal. "
    "Cliente quiere esperar o no tiene dinero ahora: 'Entiendo, no se preocupe. Este plan no requiere pago previo y el primer cobro seria en la fecha correspondiente. Tambien ofrecemos pagos en quincenas. Puede activarlo desde ahora sin pagar por adelantado. Que le parece?' Si prefiere esperar: 'Puedo enviarle la informacion por correo para que la revise con calma.' "
    "Cliente dice que prepago le da mas control: 'Claro, aunque con este plan mantiene control porque paga el mismo monto mensual, sin recargas.' "
    "Cliente no puede comprometerse a pagar: 'Gracias por escuchar la oferta, puede contactarnos cuando lo desee.' "
    "Cliente pregunta por que ahora la oferta: 'Es un plan nuevo disponible para clientes exclusivos como usted.' "
    "Solo si el cliente tuvo mala experiencia o reclamo: 'Siento mucho su experiencia. Para soporte o reclamos puede comunicarse al 5073907555. Su comentario es importante para nosotros.' "
    "Cliente sin trabajo fijo o duda de requisitos: 'Solo necesita cedula y correo para activar el plan, ademas contamos con pagos quincenales.' "
    "Volver a prepago: Debe tener cuenta al dia o en cero y solicitarlo via WhatsApp o telefono. "
    "Tiempo recomendado en el plan: Se recomienda minimo 6 meses para acceder a beneficios como equipos. "
    "Cancelacion del plan: No hay penalidad, pero pierde beneficios. "
    "Metodos de pago: App Mi Tigo, Yappy, tarjeta, banca en linea, Western Union y puntos fisicos. "
    "Pago quincenal: Puede dividir el monto mensual en dos pagos iguales. "
    "Mantener numero: Si, sin costo. Nuevo numero: Si, se puede asignar. "
    "Saldo prepago: Se aplica antes de activar. Cambio de SIM: Con otro operador si, con Tigo prepago no. "
    "Ventajas postpago vs prepago: Mas datos, minutos ilimitados, roaming y apps incluidos. "
    "Compatibilidad del equipo: Se puede verificar en el momento. Depositos o requisitos: Solo documento de identidad. "
    "Roaming incluye: Canada, Estados Unidos, Mexico, Guatemala, Honduras, El Salvador, Nicaragua, Costa Rica, Belice, Colombia, Venezuela, Ecuador, Chile, Peru, Bolivia, Paraguay, Uruguay, Brasil y Argentina. Si el pais no esta en la lista, NO esta incluido. "
    "Facturacion: Servicios hogar vs movil: Se facturan por separado con ciclos distintos. "
    "Politicas: Todos los planes tienen politica de uso justo. Puede consultarla en la web de Tigo Panama. "
    "Nota importante: Camila nunca debe decir que Tigo Security protege contra robo, permite localizacion o borrado de datos. "
)

SANITIZED_SYSTEM_PROMPT_ES = clean_system_prompt(SYSTEM_PROMPT_ES, allowed_punct={".", ","})


def get_system_prompt(lang: str | None = None) -> str:
    """Return the appropriate system prompt based on language.

    Always returns the Spanish (Tigo/Camila) prompt as the default — this agent
    is designed for Tigo Panama and must always use the Spanish system prompt.
    The prompt itself instructs the agent to always respond in Spanish regardless
    of the user's language.
    Note: custom_prompt handling is done in build_messages() — when a custom prompt is set,
    it completely replaces this default prompt and the language instruction.
    """
    return SYSTEM_PROMPT_ES


def get_sanitized_system_prompt(lang: str | None = None) -> str:
    """Return the appropriate sanitized system prompt based on language."""
    from STT_server.config import DEFAULT_CALL_LANGUAGE
    resolved = lang or DEFAULT_CALL_LANGUAGE
    if resolved == "es":
        return SANITIZED_SYSTEM_PROMPT_ES
    return SANITIZED_SYSTEM_PROMPT

# ── Spanish language markers (re-enabled — full Spanish mode) ──
SPANISH_LANGUAGE_MARKERS = (
    "hola",
    "gracias",
    "por favor",
    "buenos",
    "buenas",
    "necesito",
    "quiero",
    "puedo",
    "ayuda",
    "como",
    "donde",
    "cuanto",
)
# SPANISH_LANGUAGE_MARKERS: tuple[str, ...] = ()  # empty — Spanish detection disabled (English mode)

ENGLISH_LANGUAGE_MARKERS = (
    "hello",
    "thanks",
    "thank you",
    "please",
    "help",
    "need",
    "want",
    "where",
    "how",
    "what",
    "today",
)

INCOMPLETE_TRAILING_MARKERS = {
    "a",
    "about",
    "also",
    "an",
    "and",
    "because",
    "been",
    "but",
    "como",
    "con",
    "de",
    "del",
    "el",
    "for",
    "i",
    "if",
    "just",
    "la",
    "like",
    "los",
    "me",
    "my",
    "o",
    "or",
    "para",
    "pero",
    "please",
    "por",
    "porque",
    "que",
    "si",
    "so",
    "sobre",
    "some",
    "than",
    "that",
    "the",
    "then",
    "to",
    "with",
    "y",
    "yo",
}

INCOMPLETE_TRAILING_PHRASES = {
    "and i",
    "and my",
    "because i",
    "can you",
    "could you",
    "de mi",
    "for my",
    "i need",
    "i want",
    "me gustaria",
    "para mi",
    "por que",
    "que me",
    "y mi",
    "y yo",
}


def normalize_supported_language(lang: str | None) -> str:
    if not lang:
        return DEFAULT_CALL_LANGUAGE if DEFAULT_CALL_LANGUAGE in SUPPORTED_LANGUAGES else "es"

    lowered = lang.strip().lower()
    if lowered in SUPPORTED_LANGUAGES:
        return lowered
    if lowered in {"english", "en-us", "en-gb"} or lowered.startswith("en-"):
        return "en"
    if lowered in {"spanish", "es-419", "es-es"} or lowered.startswith("es-"):
        return "es"
    return DEFAULT_CALL_LANGUAGE if DEFAULT_CALL_LANGUAGE in SUPPORTED_LANGUAGES else "es"


def infer_supported_language_from_text(text: str, fallback: str = "es") -> str:
    # Full Spanish mode — always returns "es"
    # To re-enable English, disable SPANISH_LANGUAGE_MARKERS and change return to "en"
    return "es"
    # lowered = text.lower().strip()
    # if not lowered:
    #     return normalize_supported_language(fallback)
    #
    # english_hits = sum(marker in lowered for marker in ENGLISH_LANGUAGE_MARKERS)
    # spanish_hits = sum(marker in lowered for marker in SPANISH_LANGUAGE_MARKERS)
    # has_spanish_chars = any(char in lowered for char in "áéíóúñ¿¡")
    #
    # if has_spanish_chars or spanish_hits > english_hits:
    #     return "es"
    # if english_hits > spanish_hits:
    #     return "en"
    # return normalize_supported_language(fallback)


def detect_language(text: str) -> str:
    # Full Spanish mode — always returns "es"
    # To re-enable detection, uncomment the original line.
    return "es"
    # return infer_supported_language_from_text(text, fallback=DEFAULT_CALL_LANGUAGE)


def get_language_instruction(lang: str) -> str:
    # Full English mode — always returns English instruction.
    # To re-enable Spanish, uncomment the block below.
    return (
        "Responde solo en espanol. Maximo 1-2 frases cortas. "
        "No cambies de idioma salvo que el usuario lo haga explicitamente."
    )
    # if normalize_supported_language(lang) == "en":
    #     return (
    #         "Reply only in English. Keep responses to 1-2 short sentences. "
    #         "Do not switch language unless the user explicitly does."
    #     )
    # return (
    #     "Responde solo en espanol. Maximo 1-2 frases cortas. "
    #     "No cambies de idioma salvo que el usuario lo haga explicitamente."
    # )


def extract_structured_data(text: str) -> dict[str, str]:
    results: dict[str, str] = {}
    lowered = text.lower()

    # Normalize spoken digits before looking for order numbers.
    normalized = normalize_digits_in_text(text)

    # Order number pattern (5-6 contiguous digits) — works on normalized text.
    match = re.search(r"\b(\d{5,6})\b", normalized)
    if match:
        results["order_number"] = match.group(1)
    else:
        # Fallback: also try on original text in case normalization missed it.
        match = re.search(r"\b(\d{5,6})\b", text)
        if match:
            results["order_number"] = match.group(1)

    # Email pattern
    match = re.search(r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b", text)
    if match:
        results["email"] = match.group(1)

    # Phone number pattern
    match = re.search(r"\b(\+?\d{7,15})\b", re.sub(r"[\s().-]", "", text))
    if match:
        results["phone"] = match.group(1)

    # Name pattern (simple, case-insensitive, robust to punctuation)
    name_match = re.search(
        r"\b(?:my name is|mi nombre es)\b\s+([A-Za-zÀ-ÿ'`-]+(?:\s+[A-Za-zÀ-ÿ'`-]+){0,2})",
        text,
        flags=re.IGNORECASE,
    )
    if name_match:
        name_value = name_match.group(1).strip().strip(".?,!")
        # Remove trailing conjunction if STT merged the next clause.
        name_value = re.sub(r"\s+(?:and|y)$", "", name_value, flags=re.IGNORECASE)
        if name_value:
            results["name"] = name_value

    # Address or city request is more complex; skip for now.
    return results


def is_duplicate_collected_data(session, structured_data: dict[str, str]) -> bool:
    for key, value in structured_data.items():
        existing = session.collected_data.get(key)
        if existing and existing.lower() == value.lower():
            return True
    return False


def get_tts_model(lang: str) -> str:
    # Returns the configured ElevenLabs voice ID.
    return ELEVENLABS_TTS_VOICE_ID


def get_filler_text(lang: str) -> str:
    if not FILLER_TTS_ENABLED:
        return ""
    # Full Spanish mode — always returns Spanish filler.
    return FILLER_TEXT_ES
    # return FILLER_TEXT_EN if normalize_supported_language(lang) == "en" else FILLER_TEXT_ES


def get_stt_failure_prompt(lang: str) -> str:
    # Full Spanish mode — always returns Spanish prompt.
    return STT_FAILURE_PROMPT_ES
    # return STT_FAILURE_PROMPT_EN if normalize_supported_language(lang) == "en" else STT_FAILURE_PROMPT_ES


def normalize_deepgram_language(lang: str | None) -> str | None:
    if not lang:
        return None

    lowered = lang.strip().lower()
    if lowered in {"en", "en-us", "en-gb", "english"} or lowered.startswith("en-"):
        return "en"
    if lowered in {"es", "es-419", "es-es", "spanish"} or lowered.startswith("es-"):
        return "es"
    return None


# Greetings, fillers, name-mentions and acknowledgments that should not
# trigger an LLM response on their own.  They get deferred and merged
# with the real user request when it arrives.
NON_ACTIONABLE_PHRASES = {
    # English greetings / fillers
    "hi", "hello", "hey", "yo", "good morning", "good afternoon",
    "good evening", "good day", "how are you", "howdy",
    # Spanish greetings / fillers
    "hola", "buenos dias", "buenas tardes", "buenas noches", "buenas",
    "buenos", "que tal",
    # Agent name variations
    "tessa", "hi tessa", "hello tessa", "hey tessa", "hola tessa",
    "camila", "hola camila", "oye camila",
    # Acknowledgments / stalls
    "ok", "okay", "sure", "yes", "yeah", "yep", "no", "nah", "nope",
    "si", "vale", "wait", "hold on", "one moment", "un momento",
    "espera", "oh", "oh well", "um", "uh", "hmm", "ah", "right",
    "got it", "i see", "oh ok", "oh okay", "thanks", "thank you",
    "gracias",
    # Lone pronouns / fragments that Deepgram emits as isolated finals
    "i", "he", "she", "we", "they", "it", "you", "me", "us", "them",
    "yo", "el", "ella", "ellos", "nosotros",
    "ah bueno", "oh bueno", "bueno", "bien", "pues", "este",
    "so", "well", "like", "actually", "anyway",
}


def is_non_actionable_utterance(text: str) -> bool:
    """Return True if the text is purely a greeting / filler / acknowledgment."""
    cleaned = text.strip().lower()
    # Strip trailing punctuation for matching
    cleaned = cleaned.rstrip(".,!?;:")
    cleaned = cleaned.strip()
    if not cleaned:
        return False
    return cleaned in NON_ACTIONABLE_PHRASES


def looks_like_incomplete_utterance(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if stripped.endswith((",", ";", ":", "-", "(", "/")):
        return True

    if stripped[-1] in ".!?":
        return False

    tokens = stripped.lower().replace("?", "").replace("!", "").replace(".", "").split()
    if not tokens:
        return False

    # Digit dictation in progress — user may still be speaking digits.
    # Only treat as incomplete if fewer than 5 digits accumulated so far.
    if looks_like_digit_dictation(stripped):
        normalized = normalize_digits_in_text(stripped)
        digit_runs = re.findall(r"\d+", normalized)
        max_digits = max((len(r) for r in digit_runs), default=0)
        if max_digits < 5:
            return True

    last_token = tokens[-1]
    if last_token in INCOMPLETE_TRAILING_MARKERS:
        return True

    if len(tokens) >= 2:
        last_phrase = " ".join(tokens[-2:])
        if last_phrase in INCOMPLETE_TRAILING_PHRASES:
            return True

    return False


def split_tts_segments(text: str, max_chars: int = 300) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    segments: list[str] = []
    current: list[str] = []
    for idx, char in enumerate(stripped):
        current.append(char)
        # Solo cortar en fin de frase (., !, ?)
        if char in ".!?":
            # No cortar si el siguiente caracter es parte de la misma palabra (ej: Dr. Smith)
            next_char = stripped[idx+1] if idx+1 < len(stripped) else ""
            if next_char and next_char not in " \n\t":
                continue
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []

    # Agregar cualquier resto como segmento final
    if current:
        segment = "".join(current).strip()
        if segment:
            segments.append(segment)

    return segments


def pop_streaming_segments(buffer: str, force: bool = False) -> tuple[list[str], str]:
    remainder = buffer
    segments: list[str] = []

    while remainder:
        cut_index: int | None = None
        # First segment uses a lower threshold for faster TTFT
        is_first = len(segments) == 0
        min_punct = 5 if is_first else 15
        max_chars = STREAMING_FIRST_SEGMENT_CHARS if is_first else STREAMING_SEGMENT_MAX_CHARS

        for index, char in enumerate(remainder):
            if char in ".!?\n" and index >= min_punct:
                cut_index = index + 1
                break

        if cut_index is None and len(remainder) >= max_chars:
            cut_index = remainder.rfind(" ", 0, max_chars)
            if cut_index <= 0:
                cut_index = max_chars

        if cut_index is None:
            break

        segment = remainder[:cut_index].strip()
        remainder = remainder[cut_index:].lstrip()
        if segment:
            segments.append(segment)

    if force and remainder.strip():
        segments.append(remainder.strip())
        remainder = ""

    return segments, remainder


def sanitize_tts_text(text: str, max_len: int = 1500, allowed_punct: set[str] | None = None) -> str:
    """Sanitize text intended for TTS playback."""
    # Sanitization disabled — return the original text unchanged to avoid
    # audio corruption issues. This function remains as a no-op so callers
    # that reference it keep working without modification.
    return text
