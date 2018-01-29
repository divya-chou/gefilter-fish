// Grab the ASIN and product title
var asin         = document.getElementById("ASIN").value;
var productTitle = document.getElementById("productTitle").innerText;

console.log("Hello");

// Check if ASIN is in our database
$.get( "https://localhost:5000/in_db", { asin: asin } )
  .done( function ( in_db ) {

    console.log("In here");

    // If in the DB then modify the page
    if ( in_db ) {

      // Grab the topics and add in the buttons
      $.get( "https://localhost:5000/model", { asin: asin } )
        .done( function( data ) { 

        // Replace the title of the section
        $('h3[data-hook="lighthut-title"]').replaceWith('<h3 data-hook="lighthut-title" class="a-spacing-base">Topics in customer reviews</h3>');
     
          // Add in buttons
          $('.cr-lighthouse-terms').replaceWith('<div class="cr-lighthouse-terms">');
          for (var i = 0; i < data.topic.length; i++) {
            $('.cr-lighthouse-terms').append('<form target="_blank" action="https://localhost:5000/reviews?topic='+i+'&title='+encodeURIComponent(productTitle)+'" method="post">'+
                                               ' <button class="cr-lighthouse-term" name="topic'+i+' type="submit">'+data.topic[i]+'</button>'+
                                             '</form>');
          }
          $('.cr-lighthouse-terms').append('</div>');
          $('form').css({"display": "inline"});
          $('.cr-lighthouse-terms').css({"max-height": "200px"});
      
        });

    } 

  });




