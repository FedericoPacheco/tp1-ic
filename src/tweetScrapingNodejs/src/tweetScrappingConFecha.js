import puppeteer from 'puppeteer';
import fs from 'fs';

const eliminarObjetosRepetidosEnArr = (arr) => {
    // Usamos un nuevo array 'objetosUnicos' para almacenar los objetos únicos.
    const objetosUnicos = [];

    // Usamos un objeto Set para realizar un seguimiento de los IDs que ya hemos visto.
    const textoVisto = new Set();

    // Recorremos el array original y verificamos si el ID ya ha sido visto.
    for (const objeto of arr) {

        if (!textoVisto.has(objeto.texto)) {
            // Si el ID no ha sido visto, lo agregamos al Set y al nuevo array.
            textoVisto.add(objeto.texto);
            objetosUnicos.push(objeto);
        }
    }

    return objetosUnicos
}

const scrapping = async (query, minTweets) => {
    // Lanzamos un navegador
    const browser = await puppeteer.launch({ headless: false });

    // Creamos una pestaña
    const page = await browser.newPage();

    // Navegar hacia el sitio
    await page.goto(`https://twitter.com/search?q=${query}&f=live`);

    // Espero por el input del mail
    const inputMail = await page.waitForSelector('.r-30o5oe.r-1niwhzg.r-17gur6a.r-1yadl64.r-deolkf.r-homxoj.r-poiln3.r-7cikom.r-1ny4l3l.r-t60dpp.r-1dz5y72.r-fdjqy7.r-13qz1uu');
    // Escribo en el input
    await inputMail.type("Conan1339017", { delay: 100 });

    // Presiono enter
    await page.keyboard.press('Enter');

    // Espero por el input de password
    const inputPassword = await page.waitForSelector('.r-30o5oe.r-1niwhzg.r-17gur6a.r-1yadl64.r-deolkf.r-homxoj.r-poiln3.r-7cikom.r-1ny4l3l.r-t60dpp.r-1dz5y72.r-fdjqy7.r-13qz1uu');
    // Escribo en el input
    await inputPassword.type("mileiScraping", { delay: 100 });

    // Presiono enter
    await page.keyboard.press('Enter');

    // Espero por el contenedor de los tweets
    await page.waitForSelector('[data-testid="cellInnerDiv"]')

    //await page.waitForSelector('[data-testid="tweetText"]')

    let dataSet = []

    for (let i = 0; i < minTweets / 10; i++) {

        let tweets = await page.$$eval('div[data-testid="cellInnerDiv"]', divs => {

            return divs.map(div => {

                const time = div.querySelector('time')?.getAttribute("datetime"); // <time datetime="2023-09-05T15:41:59.000Z">1m</time>
                const texto = div.querySelector('div[data-testid="tweetText"]')?.textContent;

                return {
                    time: time ?? "",
                    texto: texto ?? "",
                }
            })

        });

        dataSet = [...dataSet, ...tweets]

        //await page.evaluate('window.scrollBy(0,2000)');
        await page.mouse.wheel({ deltaY: 3000 });
        await new Promise(r => setTimeout(r, 1500));
    }

    //console.log(dataSet)
    console.log(dataSet.length)

    // Elimino repetidos
    const dataSetConjunto = eliminarObjetosRepetidosEnArr(dataSet);
    console.log("Tweets obtenidos: " + dataSetConjunto.length)

    // Cerrar el navegador
    await browser.close();

    return dataSetConjunto
}

const armarQuery = (query) => {
    return query.replace(" ", "%20")
}

const armarCSV = (data) => {

    let tweetsCSV = "Fecha;Tweet;\n";

    data.forEach(tweetObj => {

        let tweetSinSaltoLinea = tweetObj.texto.replace(/\n/g, " ") // Quito enter
        tweetSinSaltoLinea = tweetSinSaltoLinea.replaceAll(";", ".") // Quito ;
        tweetsCSV = tweetsCSV + tweetObj.time + ";" + tweetSinSaltoLinea + ";\n" // fechadeltweet;texto sin puntos y coma ni enter; \n

    });

    fs.writeFileSync("tweets.csv", tweetsCSV);
}

const main = (async () => {

    const query = "Milei libertad"
    const minTweets = 100

    const data = await scrapping(armarQuery(query), minTweets);

    armarCSV(data)
})();
